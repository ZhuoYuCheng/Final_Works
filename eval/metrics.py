from typing import Dict, List, Tuple

import torch


def normalize_role(role: str) -> str:
    if role is None:
        return ""
    role = role.strip()
    if "->" in role:
        role = role.split("->", 1)[1].strip()
    if "(" in role:
        role = role.split("(", 1)[0].strip()
    return role


def normalize_agent_key(key: str) -> str:
    return normalize_role((key or "")).lower()


def aggregate_agent_logits(
    logits: torch.Tensor, agent_keys: List[str], mode: str = "sum", top_k: int = 0
) -> Tuple[str, torch.Tensor, List[str]]:
    key_to_vals: Dict[str, List[torch.Tensor]] = {}
    for idx, key in enumerate(agent_keys):
        norm = normalize_agent_key(key)
        if not norm:
            continue
        key_to_vals.setdefault(norm, []).append(logits[idx])
    key_logits: List[torch.Tensor] = []
    keys: List[str] = []
    for key, vals in key_to_vals.items():
        keys.append(key)
        if mode == "max":
            key_logits.append(torch.stack(vals).max())
        elif mode == "logsumexp":
            key_logits.append(torch.logsumexp(torch.stack(vals), dim=0))
        elif mode == "topk":
            k = top_k if top_k > 0 else len(vals)
            k = min(k, len(vals))
            top_vals = torch.topk(torch.stack(vals), k=k).values
            key_logits.append(top_vals.sum())
        else:
            key_logits.append(torch.stack(vals).sum())
    if not key_logits:
        return "", torch.empty(0), []
    scores = torch.stack(key_logits)
    pred_idx = int(torch.argmax(scores).item())
    return keys[pred_idx], scores, keys


def compute_metrics(pred_idx: int, pred_who: str, label_when, label_who) -> Dict[str, int]:
    out = {"when_correct": 0, "who_correct": 0, "when_total": 0, "who_total": 0}
    if label_when is not None:
        out["when_total"] = 1
        if pred_idx == label_when:
            out["when_correct"] = 1
    if label_who:
        out["who_total"] = 1
        if normalize_agent_key(pred_who) == normalize_agent_key(label_who):
            out["who_correct"] = 1
    return out
