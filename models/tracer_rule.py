import torch

from models.soft_idg import backtrace_scores


def compute_tracer_scores(
    weights: torch.Tensor,
    depth: int,
    dist_power: float,
    mode: str,
    anomaly_scores: torch.Tensor,
    anomaly_weight: float,
    start_idx: int | None = None,
) -> torch.Tensor:
    back_scores = backtrace_scores(
        weights, depth, dist_power=dist_power, mode=mode, start_idx=start_idx
    )
    if anomaly_scores.numel() == back_scores.numel() and anomaly_weight > 0:
        scores = (1 - anomaly_weight) * back_scores + anomaly_weight * anomaly_scores
    else:
        scores = back_scores
    return scores
