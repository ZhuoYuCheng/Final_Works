import argparse
import os
import random
from typing import List, Optional

import torch
import torch.nn as nn

from data.schema import UnifiedTrajectory
from datasets.registry import load_dataset, parse_subset_list
from eval.metrics import aggregate_agent_logits, compute_metrics, normalize_agent_key
from models.feature_builder import build_step_features
from models.masc_detector import compute_proto_scores
from models.slm_wrapper import SLMWrapper, serialize_steps
from models.soft_idg import (
    backtrace_scores,
    build_soft_idg,
    build_soft_idg_debug,
    build_step_token_map,
)
from models.tracer_gnn import GraphScorer, StepScorer
from models.tracer_seq import SequenceScorer
from models.tracer_rule import compute_tracer_scores


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_history(steps, mistake_idx, max_steps, mode: str):
    if max_steps is None or len(steps) <= max_steps:
        return steps, mistake_idx

    n = len(steps)
    if mode == "cover_error" and mistake_idx is not None:
        half = max_steps // 2
        start = max(0, mistake_idx - half)
        end = start + max_steps
        if end > n:
            end = n
            start = max(0, end - max_steps)
        idxs = list(range(start, end))
        sub_steps = [steps[i] for i in idxs]
        new_idx = idxs.index(mistake_idx) if mistake_idx in idxs else None
        return sub_steps, new_idx
    if mode == "head":
        idxs = list(range(max_steps))
    elif mode == "head_tail":
        head = max_steps // 2
        tail = max_steps - head
        idxs = list(range(head)) + list(range(n - tail, n))
    elif mode == "uniform":
        step = (n - 1) / float(max_steps - 1)
        idxs = [int(i * step) for i in range(max_steps)]
    else:
        idxs = list(range(n - max_steps, n))

    sub_steps = [steps[i] for i in idxs]
    new_idx = None
    if mistake_idx is not None:
        try:
            new_idx = idxs.index(mistake_idx)
        except ValueError:
            new_idx = None
    return sub_steps, new_idx


def compute_step_embeddings(hidden: torch.Tensor, step_tokens):
    embs = []
    for toks in step_tokens:
        if toks:
            embs.append(hidden[toks].mean(dim=0))
        else:
            embs.append(torch.zeros(hidden.size(1)))
    return torch.stack(embs, dim=0)


def compute_delta_scores(step_embs: torch.Tensor) -> torch.Tensor:
    if step_embs.numel() == 0:
        return torch.zeros(0)
    scores = torch.zeros(step_embs.size(0))
    if step_embs.size(0) <= 1:
        return scores
    prev = step_embs[:-1]
    cur = step_embs[1:]
    sim = nn.functional.cosine_similarity(cur, prev, dim=1)
    scores[1:] = 1.0 - sim
    return scores


def _normalize_anomaly(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    if mode == "zscore":
        mean = scores.mean()
        std = scores.std(unbiased=False)
        return (scores - mean) / (std + 1e-6)
    if mode == "minmax":
        min_v = scores.min()
        max_v = scores.max()
        return (scores - min_v) / (max_v - min_v + 1e-6)
    return scores


def compute_anomaly_scores(step_embs: torch.Tensor, args) -> torch.Tensor:
    if args.anomaly_mode == "none":
        return torch.zeros(step_embs.size(0))
    proto = compute_proto_scores(
        step_embs, mode=args.proto_mode, bank_size=args.proto_bank_size
    )
    if args.anomaly_mode == "proto":
        return _normalize_anomaly(proto, args.anomaly_norm)
    if args.anomaly_mode == "proto_delta":
        delta = compute_delta_scores(step_embs)
        return _normalize_anomaly(0.5 * (proto + delta), args.anomaly_norm)
    if args.anomaly_mode == "proto_prefix":
        return _normalize_anomaly(
            compute_proto_scores(step_embs, mode="prefix"), args.anomaly_norm
        )
    if args.anomaly_mode == "proto_bank":
        return _normalize_anomaly(
            compute_proto_scores(step_embs, mode="bank", bank_size=args.proto_bank_size),
            args.anomaly_norm,
        )
    return _normalize_anomaly(proto, args.anomaly_norm)


def build_embed_features(
    step_embs: torch.Tensor, scores: torch.Tensor, embed_extra: str
) -> torch.Tensor:
    if embed_extra == "none":
        return step_embs
    extras = []
    if "scores" in embed_extra:
        extras.append(scores.unsqueeze(1))
    if "pos" in embed_extra:
        t = step_embs.size(0)
        pos = torch.arange(t, dtype=torch.float32) / max(t - 1, 1)
        extras.append(pos.unsqueeze(1))
    if not extras:
        return step_embs
    return torch.cat([step_embs, *extras], dim=1)


def apply_time_prior(vec: torch.Tensor, weight: float, power: float) -> torch.Tensor:
    if weight == 0 or vec.numel() == 0:
        return vec
    denom = max(vec.numel() - 1, 1)
    pos = torch.arange(vec.numel(), device=vec.device, dtype=vec.dtype) / denom
    prior = (1.0 - pos).pow(power)
    return vec + weight * prior


def find_first_error_step(steps) -> Optional[int]:
    keywords = [
        "error",
        "failed",
        "failure",
        "incorrect",
        "wrong",
        "cannot",
        "unable",
        "exception",
        "traceback",
        "invalid",
        "not found",
        "sorry",
        "apolog",
    ]
    for idx, step in enumerate(steps):
        text = (step.output or "").lower()
        if any(k in text for k in keywords):
            return idx
    return None


def select_failure_step(steps, anomaly: torch.Tensor, strategy: str) -> int:
    if not steps:
        return 0
    if strategy == "max_anomaly" and anomaly.numel() > 0:
        return int(torch.argmax(anomaly).item())
    if strategy == "first_error":
        idx = find_first_error_step(steps)
        if idx is not None:
            return idx
    return len(steps) - 1


def resolve_failure_sinks(steps, anomaly: torch.Tensor, args):
    sinks = []
    parts = [p.strip() for p in args.failure_sinks.split(",") if p.strip()]
    for name in parts:
        if name == "final":
            sinks.append((len(steps) - 1, args.sink_weight_final))
        elif name == "max_anomaly":
            if anomaly.numel() > 0:
                idx = int(torch.argmax(anomaly).item())
                weight = float(anomaly[idx].item()) if args.sink_weight_anomaly <= 0 else args.sink_weight_anomaly
                sinks.append((idx, weight))
        elif name == "first_error":
            idx = find_first_error_step(steps)
            if idx is not None:
                sinks.append((idx, args.sink_weight_first_error))
    if not sinks:
        sinks = [(len(steps) - 1, 1.0)]
    total = sum(max(w, 0.0) for _, w in sinks)
    if total <= 0:
        return [(idx, 1.0 / len(sinks)) for idx, _ in sinks]
    return [(idx, w / total) for idx, w in sinks]



def extract_greedy_path(weights: torch.Tensor, start_idx: int, max_hops: int = 10) -> List[int]:
    path = [start_idx]
    current = start_idx
    for _ in range(max_hops):
        if current <= 0:
            break
        incoming = weights[:current, current]
        if incoming.numel() == 0:
            break
        best = int(torch.argmax(incoming).item())
        if best in path:
            break
        path.append(best)
        current = best
    return path


def top_incoming_edges(weights: torch.Tensor, node: int, top_n: int = 3):
    if node <= 0:
        return []
    vals = weights[:node, node]
    if vals.numel() == 0:
        return []
    top_n = min(top_n, vals.numel())
    scores, idxs = torch.topk(vals, k=top_n)
    return [(int(idx.item()), float(score.item())) for idx, score in zip(idxs, scores)]


def collect_incoming_neighbors(weights: torch.Tensor, start_idx: int, hops: int) -> List[int]:
    if weights.numel() == 0:
        return [start_idx]
    visited = {start_idx}
    frontier = [start_idx]
    for _ in range(max(hops, 0)):
        next_frontier = []
        for node in frontier:
            if node <= 0:
                continue
            incoming = weights[:node, node]
            if incoming.numel() == 0:
                continue
            srcs = (incoming > 0).nonzero(as_tuple=False).view(-1).tolist()
            for src in srcs:
                if src not in visited:
                    visited.add(src)
                    next_frontier.append(src)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


def write_diagnostic_report(
    cache,
    pred_idx,
    pred_agent,
    args,
    report_dir,
    tag,
):
    os.makedirs(report_dir, exist_ok=True)
    failure_idx = cache.get("failure_idx", len(cache["history"]) - 1)
    weights = cache["weights"]
    path = extract_greedy_path(weights, failure_idx, max_hops=10)
    raw = cache.get("raw_weights")
    pruned_edges = cache.get("pruned_edges", [])

    report_path = os.path.join(report_dir, f"{tag}.txt")
    dot_path = os.path.join(report_dir, f"{tag}.dot")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"failure_idx: {failure_idx}\n")
        f.write(f"failure_strategy: {args.failure_step_strategy}\n")
        f.write(f"pred_when: {pred_idx}\n")
        f.write(f"pred_who: {pred_agent}\n")
        f.write(f"evidence_path: {list(reversed(path))}\n\n")

        f.write("top-incoming edges per path node (kept):\n")
        for node in reversed(path):
            edges = top_incoming_edges(weights, node, top_n=args.diagnose_top_n)
            f.write(f"  node {node}: {edges}\n")

        if pruned_edges:
            f.write("\npruned edges (top by raw weight):\n")
            for i, j, w in pruned_edges[: args.diagnose_top_n * 3]:
                f.write(f"  {i}->{j}: {w:.6f}\n")
        else:
            f.write("\npruned edges: none or debug disabled\n")

    # DOT graph: include path nodes and their top incoming edges
    nodes = set(path)
    edge_lines = []
    for node in reversed(path):
        for src, w in top_incoming_edges(weights, node, top_n=args.diagnose_top_n):
            nodes.add(src)
            color = "red" if src in path and node in path else "gray"
            edge_lines.append(f'  {src} -> {node} [label=\"{w:.3f}\", color={color}];')
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph G {\n  rankdir=LR;\n")
        for node in sorted(nodes):
            label = f"{node}" if node != failure_idx else f"{node} (failure)"
            f.write(f'  {node} [label=\"{label}\"];\n')
        for line in edge_lines:
            f.write(f"{line}\n")
        f.write("}\n")

    json_path = os.path.join(report_dir, f"{tag}.json")
    payload = {
        "failure_idx": failure_idx,
        "failure_strategy": args.failure_step_strategy,
        "pred_when": pred_idx,
        "pred_who": pred_agent,
        "pruned_edges": pruned_edges[: args.diagnose_top_n * 3],
    }
    try:
        import json

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_caches(records: List[UnifiedTrajectory], extractor: SLMWrapper, args, train_mode: bool = False):
    caches = []
    for rec in records:
        steps, mistake_idx = sample_history(
            rec.steps, rec.labels.when, args.max_steps, args.history_sample
        )
        if not steps:
            continue
        if train_mode and args.train_prefix_only and mistake_idx is not None and mistake_idx > 0:
            cut = min(mistake_idx + 1, len(steps))
            steps = steps[:cut]

        text, spans = serialize_steps(steps, max_chars_per_step=args.max_chars_per_step)
        attn_matrix, offsets, hidden = extractor.encode_text(text)
        step_tokens = build_step_token_map(offsets, spans)
        step_embs = compute_step_embeddings(hidden, step_tokens)
        final_emb = step_embs[-1] if step_embs.numel() > 0 else torch.zeros(1)
        sim = nn.functional.cosine_similarity(step_embs, final_emb.unsqueeze(0), dim=1)
        dist = (step_embs - final_emb).pow(2).sum(dim=1).sqrt()
        dist = dist / max(dist.max().item(), 1.0)

        if args.diagnose_k > 0:
            weights, debug = build_soft_idg_debug(
                attn_matrix,
                step_tokens,
                top_k=args.top_k,
                min_edge_weight=args.min_edge_weight,
                keep_ratio=args.keep_ratio,
                prune_mode=args.prune_mode,
                zscore_k=args.zscore_k,
                renorm_mode=args.renorm_mode,
                edge_temperature=args.edge_temperature,
                min_in_edges=args.min_in_edges,
                normalize_by_src_len=args.normalize_by_src_len,
                token_agg=args.token_agg,
                token_topk=args.token_topk,
                step_texts=[s.output for s in steps],
                overlap_weight=args.overlap_weight,
            )
        else:
            weights = build_soft_idg(
                attn_matrix,
                step_tokens,
                top_k=args.top_k,
                min_edge_weight=args.min_edge_weight,
                keep_ratio=args.keep_ratio,
                prune_mode=args.prune_mode,
                zscore_k=args.zscore_k,
                renorm_mode=args.renorm_mode,
                edge_temperature=args.edge_temperature,
                min_in_edges=args.min_in_edges,
                normalize_by_src_len=args.normalize_by_src_len,
                token_agg=args.token_agg,
                token_topk=args.token_topk,
                step_texts=[s.output for s in steps],
                overlap_weight=args.overlap_weight,
            )
            debug = {}

        agent_keys = [s.name or s.role for s in steps]
        anomaly = compute_anomaly_scores(step_embs, args)

        if args.failure_step_strategy == "multi":
            sinks = resolve_failure_sinks(steps, anomaly, args)
            score_sum = torch.zeros(weights.size(0))
            for idx, w in sinks:
                score_sum = score_sum + w * backtrace_scores(
                    weights,
                    depth=args.depth,
                    dist_power=args.dist_power,
                    mode=args.backtrace_mode,
                    start_idx=idx,
                )
            if anomaly.numel() == score_sum.numel() and args.anomaly_weight > 0:
                scores = (1 - args.anomaly_weight) * score_sum + args.anomaly_weight * anomaly
            else:
                scores = score_sum
            failure_idx = sinks[0][0]
        else:
            failure_idx = select_failure_step(steps, anomaly, args.failure_step_strategy)
            scores = compute_tracer_scores(
                weights,
                depth=args.depth,
                dist_power=args.dist_power,
                mode=args.backtrace_mode,
                anomaly_scores=anomaly,
                anomaly_weight=args.anomaly_weight,
                start_idx=failure_idx,
            )

        feats = build_step_features(
            weights,
            scores,
            step_tokens,
            [s.role for s in steps],
            [s.agent_id for s in steps],
            len(rec.agents),
            [s.output for s in steps],
            args.agent_feat_dim,
            sim,
            dist,
            anomaly,
        )

        caches.append(
            {
                "features": feats,
                "weights": weights,
                "scores": scores,
                "history": steps,
                "agent_keys": agent_keys,
                "mistake_idx": mistake_idx,
                "mistake_agent": rec.labels.who,
                "failure_idx": failure_idx,
                "anomaly": anomaly,
                "raw_weights": debug.get("raw_weights"),
                "pruned_edges": debug.get("pruned_edges", []),
                "step_embs": step_embs,
                "sim": sim,
            }
        )
    return caches


def train_model(train_caches, args):
    if args.train_epochs <= 0:
        return None
    if not train_caches:
        return None

    in_dim = train_caches[0]["features"].size(1)
    if args.model_type == "gnn":
        model = GraphScorer(in_dim, args.hidden_dim).to(args.device)
    elif args.model_type == "seq":
        feat_dim = build_embed_features(
            train_caches[0]["step_embs"], train_caches[0]["scores"], args.embed_extra
        ).size(1)
        model = SequenceScorer(
            feat_dim,
            hidden_dim=args.seq_hidden_dim,
            nhead=args.seq_heads,
            num_layers=args.seq_layers,
        ).to(args.device)
    else:
        model = StepScorer(in_dim, args.hidden_dim).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.train_lr)

    model.train()
    for epoch in range(args.train_epochs):
        total = 0.0
        count = 0
        for cache in train_caches:
            m = cache["mistake_idx"]
            if m is None:
                continue
            if args.model_type == "gnn":
                feats = cache["features"].to(args.device)
                logits = model(feats, cache["weights"].to(args.device))
            elif args.model_type == "seq":
                feats = build_embed_features(
                    cache["step_embs"], cache["scores"], args.embed_extra
                ).to(args.device)
                logits = model(feats.float())
            else:
                feats = cache["features"].to(args.device)
                logits = model(feats)
            loss = nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([m], device=args.device)
            )
            if args.agent_loss_weight > 0:
                pred_key, role_logits, role_keys = aggregate_agent_logits(
                    logits, cache["agent_keys"], mode="sum"
                )
                gt_key = normalize_agent_key(cache["mistake_agent"])
                if gt_key and gt_key in role_keys:
                    gt = role_keys.index(gt_key)
                    agent_loss = nn.functional.cross_entropy(
                        role_logits.unsqueeze(0), torch.tensor([gt], device=args.device)
                    )
                    loss = loss + args.agent_loss_weight * agent_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        avg = total / max(count, 1)
        print(f"train_epoch {epoch+1} loss {avg:.6f}")

    return model


def evaluate(caches, model, args, extractor=None):
    totals = {"when_correct": 0, "who_correct": 0, "when_total": 0, "who_total": 0}
    diag_written = 0

    for cache in caches:
        feats = cache["features"].to(args.device)
        if model is None:
            logits = cache["scores"].to(args.device)
        else:
            if args.model_type == "gnn":
                logits = model(feats, cache["weights"].to(args.device))
            elif args.model_type == "seq":
                seq_feats = build_embed_features(
                    cache["step_embs"], cache["scores"], args.embed_extra
                ).to(args.device)
                logits = model(seq_feats.float())
            else:
                logits = model(feats)
            if args.score_mix != 0:
                logits = logits + args.score_mix * cache["scores"].to(args.device)

        logits = apply_time_prior(logits, args.early_bias_weight, args.early_bias_power)
        pred_idx = int(torch.argmax(logits).item()) if logits.numel() else 0
        pred_idx = max(0, min(pred_idx + args.pred_offset, len(cache["history"]) - 1))

        if args.step_pred_mode == "scores":
            vec = cache["scores"]
            pred_idx = int(torch.argmax(vec[:-1]).item()) if vec.numel() > 1 else 0
        elif args.step_pred_mode == "sim":
            vec = cache["sim"]
            pred_idx = int(torch.argmax(vec[:-1]).item()) if vec.numel() > 1 else 0
        pred_idx = max(0, min(pred_idx + args.pred_offset, len(cache["history"]) - 1))

        pred_agent = None
        if args.step_pred_mode == "agent_first":
            pred_agent, _, _ = aggregate_agent_logits(
                logits, cache["agent_keys"], mode=args.agent_agg, top_k=args.agent_top_k
            )
            indices = [
                i
                for i, k in enumerate(cache["agent_keys"])
                if normalize_agent_key(k) == pred_agent
            ]
            if indices:
                best_local = max(indices, key=lambda i: float(logits[i].item()))
                pred_idx = best_local
        if pred_agent is None:
            pred_agent, _, _ = aggregate_agent_logits(
                logits, cache["agent_keys"], mode=args.agent_agg, top_k=args.agent_top_k
            )
        if args.who_mode == "neighborhood":
            mask_idx = collect_incoming_neighbors(
                cache["weights"], pred_idx, args.who_hops
            )
            masked_logits = logits.clone()
            keep = torch.zeros_like(masked_logits, dtype=torch.bool)
            keep[mask_idx] = True
            masked_logits[~keep] = -1e9
            pred_agent, _, _ = aggregate_agent_logits(
                masked_logits,
                cache["agent_keys"],
                mode=args.agent_agg,
                top_k=args.agent_top_k,
            )

        metrics = compute_metrics(
            pred_idx, pred_agent, cache["mistake_idx"], cache["mistake_agent"]
        )
        for key in totals:
            totals[key] += metrics[key]

        if (
            args.diagnose_k > 0
            and diag_written < args.diagnose_k
            and cache.get("mistake_idx") is not None
            and pred_idx != cache["mistake_idx"]
        ):
            tag = f"{args.subset}_fail_{diag_written}"
            write_diagnostic_report(
                cache,
                pred_idx,
                pred_agent,
                args,
                args.diagnose_dir,
                tag,
            )
            diag_written += 1

    when_acc = totals["when_correct"] / max(totals["when_total"], 1)
    who_acc = totals["who_correct"] / max(totals["who_total"], 1)
    print(f"when_acc {when_acc:.4f} ({totals['when_correct']}/{totals['when_total']})")
    print(f"who_acc {who_acc:.4f} ({totals['who_correct']}/{totals['who_total']})")
    return when_acc, who_acc


def main():
    parser = argparse.ArgumentParser(description="White-box fault attribution (Who/When)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--subset", default="Hand-Crafted")
    parser.add_argument("--train_subset", default=None)
    parser.add_argument("--test_subset", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--history_sample",
        choices=["tail", "head", "head_tail", "uniform", "cover_error"],
        default="tail",
    )
    parser.add_argument("--max_chars_per_step", type=int, default=600)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--head_agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--head_select", choices=["all", "top_entropy"], default="all")
    parser.add_argument("--head_top_k", type=int, default=8)
    parser.add_argument("--layer_weighting", choices=["uniform", "late", "early"], default="uniform")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_edge_weight", type=float, default=None)
    parser.add_argument("--keep_ratio", type=float, default=None)
    parser.add_argument(
        "--prune_mode", choices=["fixed", "quantile", "zscore", "soft"], default="fixed"
    )
    parser.add_argument("--zscore_k", type=float, default=1.0)
    parser.add_argument("--min_in_edges", type=int, default=1)
    parser.add_argument("--renorm", action="store_true")
    parser.add_argument("--renorm_mode", choices=["none", "l1", "softmax"], default=None)
    parser.add_argument("--edge_temperature", type=float, default=1.0)
    parser.add_argument("--normalize_by_src_len", action="store_true")
    parser.add_argument("--token_agg", choices=["mean", "max", "topk"], default="mean")
    parser.add_argument("--token_topk", type=int, default=5)
    parser.add_argument("--backtrace_mode", choices=["sum", "max"], default="sum")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--dist_power", type=float, default=0.0)

    parser.add_argument(
        "--anomaly_mode",
        choices=["none", "proto", "proto_delta", "proto_prefix", "proto_bank"],
        default="proto",
    )
    parser.add_argument("--proto_mode", choices=["global", "prefix", "bank"], default="global")
    parser.add_argument("--proto_bank_size", type=int, default=4)
    parser.add_argument("--anomaly_weight", type=float, default=0.1)
    parser.add_argument("--anomaly_norm", choices=["none", "zscore", "minmax"], default="zscore")
    parser.add_argument("--score_mix", type=float, default=0.0)
    parser.add_argument("--overlap_weight", type=float, default=0.0)
    parser.add_argument(
        "--step_pred_mode",
        choices=["model", "scores", "sim", "agent_first"],
        default="model",
    )
    parser.add_argument(
        "--agent_agg", choices=["sum", "logsumexp", "max", "topk"], default="sum"
    )
    parser.add_argument("--agent_top_k", type=int, default=3)
    parser.add_argument(
        "--failure_step_strategy",
        choices=["final", "max_anomaly", "first_error", "multi"],
        default="final",
    )
    parser.add_argument(
        "--failure_sinks",
        default="final,max_anomaly,first_error",
    )
    parser.add_argument("--sink_weight_final", type=float, default=1.0)
    parser.add_argument("--sink_weight_first_error", type=float, default=1.0)
    parser.add_argument("--sink_weight_anomaly", type=float, default=0.0)
    parser.add_argument("--train_prefix_only", action="store_true")
    parser.add_argument(
        "--who_mode",
        choices=["global", "neighborhood"],
        default="global",
    )
    parser.add_argument("--who_hops", type=int, default=2)
    parser.add_argument("--diagnose_k", type=int, default=0)
    parser.add_argument(
        "--diagnose_dir", default="/root/autodl-tmp/Attribution_Final/experiments/diagnostics"
    )
    parser.add_argument("--diagnose_top_n", type=int, default=3)

    parser.add_argument("--train_epochs", type=int, default=0)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--split_strategy", choices=["random", "deterministic"], default="random")
    parser.add_argument("--model_type", choices=["mlp", "gnn", "seq"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--agent_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--embed_extra",
        choices=["auto", "none", "scores", "scores_pos"],
        default="auto",
    )
    parser.add_argument("--seq_hidden_dim", type=int, default=256)
    parser.add_argument("--seq_heads", type=int, default=4)
    parser.add_argument("--seq_layers", type=int, default=2)
    parser.add_argument("--early_bias_weight", type=float, default=0.0)
    parser.add_argument("--early_bias_power", type=float, default=1.0)
    parser.add_argument("--pred_offset", type=int, default=0)
    parser.add_argument("--agent_feat_dim", type=int, default=0)
    parser.add_argument("--agent_feat_dim_cap", type=int, default=32)
    args = parser.parse_args()

    try:
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        args.device = "cpu"

    set_seed(args.seed)

    if args.torch_dtype == "float16":
        dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = None

    if args.embed_extra == "auto":
        args.embed_extra = "scores_pos" if args.model_type == "seq" else "none"

    extractor = SLMWrapper(
        model_name_or_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        last_n_layers=args.last_n_layers,
        cache_dir=args.model_cache_dir,
        torch_dtype=dtype,
        head_agg=args.head_agg,
        head_select=args.head_select,
        head_top_k=args.head_top_k,
        layer_weighting=args.layer_weighting,
    )

    if args.train_subset or args.test_subset:
        train_subsets = parse_subset_list(args.train_subset, [args.subset])
        test_subsets = parse_subset_list(args.test_subset, [args.subset])
        train_records = []
        for subset in train_subsets:
            train_records.extend(load_dataset(args.data_dir, subset))
        test_records = {subset: load_dataset(args.data_dir, subset) for subset in test_subsets}
    else:
        records = load_dataset(args.data_dir, args.subset)
        if args.split_strategy == "deterministic":
            records = sorted(records, key=lambda x: x.trace_id)
        else:
            split_seed = args.seed if args.split_seed is None else args.split_seed
            rnd = random.Random(split_seed)
            rnd.shuffle(records)
        split = int(len(records) * args.split_ratio)
        train_records = records[:split]
        test_records = {"mixed": records[split:]}

    if args.agent_feat_dim <= 0:
        all_recs = list(train_records)
        for subset_recs in test_records.values():
            all_recs.extend(subset_recs)
        max_agents = max((len(r.agents) for r in all_recs), default=1)
        args.agent_feat_dim = max(2, min(max_agents, args.agent_feat_dim_cap))

    if args.renorm_mode is None:
        args.renorm_mode = "l1" if args.renorm else "none"
    if args.min_edge_weight is None:
        args.min_edge_weight = args.threshold
    train_caches = build_caches(train_records, extractor, args, train_mode=True)
    model = train_model(train_caches, args)

    for subset_name, subset_records in test_records.items():
        test_caches = build_caches(subset_records, extractor, args, train_mode=False)
        print(f"[Eval] subset={subset_name} ({len(test_caches)})")
        evaluate(test_caches, model, args, extractor=extractor)


if __name__ == "__main__":
    main()
