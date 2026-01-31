import re
from typing import List, Optional, Tuple

import torch


def build_step_token_map(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]):
    step_tokens = [[] for _ in spans]
    for idx, (start, _end) in enumerate(offsets):
        if start == _end:
            continue
        for step_idx, (s_start, s_end) in enumerate(spans):
            if start >= s_start and start < s_end:
                step_tokens[step_idx].append(idx)
                break
    return step_tokens


def _tokenize(text: str) -> List[str]:
    parts = re.split(r"[^A-Za-z0-9]+", (text or "").lower())
    return [p for p in parts if len(p) > 1]


def _overlap_score(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    union = ta.union(tb)
    return len(inter) / max(len(union), 1)


def sparsify_weights(
    weights: torch.Tensor,
    top_k: Optional[int] = None,
    min_edge_weight: Optional[float] = None,
    keep_ratio: Optional[float] = None,
    prune_mode: str = "fixed",
    zscore_k: float = 1.0,
    renorm_mode: str = "l1",
    edge_temperature: float = 1.0,
    min_in_edges: int = 1,
) -> torch.Tensor:
    t = weights.size(0)
    pruned = torch.zeros_like(weights)
    for j in range(t):
        candidates = []
        for i in range(j):
            w = weights[i, j].item()
            if min_edge_weight is not None and w < min_edge_weight:
                continue
            if w > 0:
                candidates.append((i, w))
        if not candidates:
            # fallback: keep at least one strongest incoming edge
            if j > 0:
                vals = weights[:j, j]
                if vals.numel() > 0 and vals.max().item() > 0:
                    best = int(torch.argmax(vals).item())
                    candidates = [(best, float(vals[best].item()))]
            if not candidates:
                continue
        if prune_mode == "soft":
            ratio = keep_ratio if keep_ratio is not None else 0.9
            vals = torch.tensor([w for _, w in candidates], dtype=torch.float32)
            if vals.numel():
                q = torch.quantile(vals, ratio)
                softened = []
                for i, w in candidates:
                    new_w = max(0.0, w - q.item())
                    softened.append((i, new_w))
                candidates = softened
                if all(w == 0 for _, w in candidates):
                    best = max(softened, key=lambda x: x[1], default=None)
                    if best is not None:
                        candidates = [best]
        elif prune_mode == "quantile":
            ratio = keep_ratio if keep_ratio is not None else 0.9
            if ratio > 0 and ratio < 1:
                keep_n = max(1, int(len(candidates) * ratio))
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:keep_n]
        elif prune_mode == "zscore":
            vals = torch.tensor([w for _, w in candidates], dtype=torch.float32)
            mean = vals.mean().item() if vals.numel() else 0.0
            std = vals.std(unbiased=False).item() if vals.numel() else 0.0
            thresh = mean + zscore_k * std
            kept = [(i, w) for i, w in candidates if w >= thresh]
            if not kept:
                kept = sorted(candidates, key=lambda x: x[1], reverse=True)[:1]
            candidates = kept
        elif keep_ratio is not None and keep_ratio > 0 and keep_ratio < 1:
            keep_n = max(1, int(len(candidates) * keep_ratio))
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:keep_n]
        if top_k is not None and top_k > 0:
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        # ensure minimum incoming edges
        if min_in_edges > 0 and len(candidates) < min_in_edges and j > 0:
            vals = weights[:j, j]
            order = torch.argsort(vals, descending=True)
            for idx in order.tolist():
                w = float(vals[idx].item())
                if w <= 0:
                    break
                if any(i == idx for i, _ in candidates):
                    continue
                candidates.append((idx, w))
                if len(candidates) >= min_in_edges:
                    break
        for i, w in candidates:
            pruned[i, j] = w
        if renorm_mode == "l1":
            s = pruned[:j, j].sum()
            if s > 0:
                pruned[:j, j] = pruned[:j, j] / s
        elif renorm_mode == "softmax":
            vals = pruned[:j, j]
            if vals.numel() > 0:
                tau = max(edge_temperature, 1e-6)
                pruned[:j, j] = torch.softmax(vals / tau, dim=0)
    return pruned


def _compute_raw_weights(
    attn_matrix: torch.Tensor,
    step_tokens: List[List[int]],
    normalize_by_src_len: bool = False,
    token_agg: str = "mean",
    token_topk: int = 5,
    step_texts: Optional[List[str]] = None,
    overlap_weight: float = 0.0,
) -> torch.Tensor:
    t = len(step_tokens)
    weights = torch.zeros((t, t))
    overlap = None
    if step_texts is not None and overlap_weight > 0:
        overlap = torch.zeros((t, t))
        for j in range(t):
            for i in range(j):
                overlap[i, j] = _overlap_score(step_texts[i], step_texts[j])
    for j in range(t):
        tok_j = step_tokens[j]
        if not tok_j:
            continue
        attn_j = attn_matrix[tok_j]
        for i in range(j):
            tok_i = step_tokens[i]
            if not tok_i:
                continue
            mass = attn_j[:, tok_i].sum(dim=1)
            if token_agg == "max":
                w = mass.max().item()
            elif token_agg == "topk":
                k = min(token_topk, mass.numel())
                w = mass.topk(k).values.mean().item()
            else:
                w = mass.mean().item()
            if normalize_by_src_len:
                w = w / max(len(tok_i), 1)
            if overlap is not None:
                w = (1 - overlap_weight) * w + overlap_weight * overlap[i, j].item()
            weights[i, j] = w
    return weights


def build_soft_idg(
    attn_matrix: torch.Tensor,
    step_tokens: List[List[int]],
    top_k: Optional[int],
    min_edge_weight: Optional[float],
    keep_ratio: Optional[float],
    prune_mode: str,
    zscore_k: float,
    renorm_mode: str,
    edge_temperature: float,
    min_in_edges: int,
    normalize_by_src_len: bool = False,
    token_agg: str = "mean",
    token_topk: int = 5,
    step_texts: Optional[List[str]] = None,
    overlap_weight: float = 0.0,
) -> torch.Tensor:
    weights = _compute_raw_weights(
        attn_matrix,
        step_tokens,
        normalize_by_src_len=normalize_by_src_len,
        token_agg=token_agg,
        token_topk=token_topk,
        step_texts=step_texts,
        overlap_weight=overlap_weight,
    )
    return sparsify_weights(
        weights,
        top_k=top_k,
        min_edge_weight=min_edge_weight,
        keep_ratio=keep_ratio,
        prune_mode=prune_mode,
        zscore_k=zscore_k,
        renorm_mode=renorm_mode,
        edge_temperature=edge_temperature,
        min_in_edges=min_in_edges,
    )


def build_soft_idg_debug(
    attn_matrix: torch.Tensor,
    step_tokens: List[List[int]],
    top_k: Optional[int],
    min_edge_weight: Optional[float],
    keep_ratio: Optional[float],
    prune_mode: str,
    zscore_k: float,
    renorm_mode: str,
    edge_temperature: float,
    min_in_edges: int,
    normalize_by_src_len: bool = False,
    token_agg: str = "mean",
    token_topk: int = 5,
    step_texts: Optional[List[str]] = None,
    overlap_weight: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    raw = _compute_raw_weights(
        attn_matrix,
        step_tokens,
        normalize_by_src_len=normalize_by_src_len,
        token_agg=token_agg,
        token_topk=token_topk,
        step_texts=step_texts,
        overlap_weight=overlap_weight,
    )
    pruned = sparsify_weights(
        raw,
        top_k=top_k,
        min_edge_weight=min_edge_weight,
        keep_ratio=keep_ratio,
        prune_mode=prune_mode,
        zscore_k=zscore_k,
        renorm_mode=renorm_mode,
        edge_temperature=edge_temperature,
        min_in_edges=min_in_edges,
    )
    pruned_edges = []
    t = raw.size(0)
    for j in range(t):
        for i in range(j):
            w = raw[i, j].item()
            if w > 0 and pruned[i, j].item() == 0:
                pruned_edges.append((i, j, w))
    pruned_edges = sorted(pruned_edges, key=lambda x: x[2], reverse=True)
    debug = {"raw_weights": raw, "pruned_edges": pruned_edges}
    return pruned, debug


def backtrace_scores(
    weights: torch.Tensor,
    depth: Optional[int],
    dist_power: float = 0.0,
    mode: str = "sum",
    start_idx: Optional[int] = None,
) -> torch.Tensor:
    t = weights.size(0)
    scores = torch.zeros(t)
    if t == 0:
        return scores
    t_f = t - 1 if start_idx is None else max(0, min(start_idx, t - 1))
    scores[t_f] = 1.0
    for i in range(t_f - 1, -1, -1):
        if depth is None or depth <= 0:
            j_max = t_f
        else:
            j_max = min(t_f, i + depth)
        if j_max <= i:
            continue
        contrib = None
        for j in range(i + 1, j_max + 1):
            w = weights[i, j]
            if dist_power != 0:
                w = w * ((j - i) ** dist_power)
            val = w * scores[j]
            if mode == "max":
                contrib = val if contrib is None else max(contrib, val)
            else:
                contrib = val if contrib is None else contrib + val
        scores[i] = contrib if contrib is not None else 0.0
    return scores
