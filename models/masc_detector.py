import math
import torch
import torch.nn as nn


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(0)
    return nn.functional.cosine_similarity(a, b, dim=1)


def compute_proto_scores(
    step_embs: torch.Tensor, mode: str = "global", bank_size: int = 4
) -> torch.Tensor:
    if step_embs.numel() == 0:
        return torch.zeros(0)
    t = step_embs.size(0)
    if t == 1:
        return torch.zeros(1)

    if mode == "prefix":
        scores = torch.zeros(t)
        for idx in range(1, t):
            proto = step_embs[:idx].mean(dim=0)
            sim = _safe_cosine(step_embs[idx : idx + 1], proto.unsqueeze(0))
            scores[idx] = 1.0 - sim.item()
        return scores

    if mode == "bank":
        k = max(2, min(bank_size, t))
        chunk = int(math.ceil(t / k))
        protos = []
        for i in range(0, t, chunk):
            proto = step_embs[i : i + chunk].mean(dim=0)
            protos.append(proto)
        proto_bank = torch.stack(protos, dim=0)
        sims = []
        for proto in proto_bank:
            sim = _safe_cosine(step_embs, proto.unsqueeze(0))
            sims.append(sim)
        max_sim = torch.stack(sims, dim=1).max(dim=1).values
        return 1.0 - max_sim

    # global
    proto = step_embs.mean(dim=0)
    sim = _safe_cosine(step_embs, proto.unsqueeze(0))
    return 1.0 - sim
