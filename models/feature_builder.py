from typing import List

import torch


def build_step_features(
    weights: torch.Tensor,
    scores: torch.Tensor,
    step_tokens,
    roles: List[str],
    agent_ids: List[int],
    num_agents: int,
    step_texts: List[str],
    agent_feat_dim: int,
    sim_to_final: torch.Tensor,
    dist_to_final: torch.Tensor,
    anomaly: torch.Tensor,
) -> torch.Tensor:
    t = weights.size(0)
    if t == 0:
        return torch.empty((0, 0))

    in_sum = weights.sum(dim=0)
    out_sum = weights.sum(dim=1)
    in_deg = (weights > 0).sum(dim=0).to(torch.float32)
    out_deg = (weights > 0).sum(dim=1).to(torch.float32)
    pos = torch.arange(t, dtype=torch.float32) / max(t - 1, 1)
    tok_lens = torch.tensor([len(x) for x in step_tokens], dtype=torch.float32)
    tok_lens = tok_lens / max(tok_lens.max().item(), 1.0)
    in_deg = in_deg / max(in_deg.max().item(), 1.0)
    out_deg = out_deg / max(out_deg.max().item(), 1.0)

    role_flags = []
    for role in roles:
        r = (role or "").lower()
        role_flags.append(
            [
                1.0 if "websurfer" in r else 0.0,
                1.0 if "orchestrator" in r else 0.0,
                1.0 if "assistant" in r else 0.0,
                1.0 if "filesurfer" in r else 0.0,
            ]
        )
    role_flags = torch.tensor(role_flags, dtype=torch.float32)

    # dynamic agent encoding with fixed dimension
    agent_feat_dim = max(int(agent_feat_dim), 1)
    agent_onehot = torch.zeros((t, agent_feat_dim), dtype=torch.float32)
    for idx, a_id in enumerate(agent_ids):
        bucket = min(max(int(a_id), 0), agent_feat_dim - 1)
        agent_onehot[idx, bucket] = 1.0
    agent_norm = torch.tensor(agent_ids, dtype=torch.float32)
    denom = max(num_agents - 1, 1)
    agent_norm = (agent_norm / denom).unsqueeze(1)

    # lightweight tool/event proxy features from text
    tool_flags = []
    for text in step_texts:
        lower = (text or "").lower()
        has_url = 1.0 if ("http://" in lower or "https://" in lower or "www." in lower) else 0.0
        has_search = 1.0 if ("search" in lower or "query" in lower) else 0.0
        tool_flags.append([has_url, has_search])
    tool_flags = torch.tensor(tool_flags, dtype=torch.float32)

    feats = torch.stack([scores, in_sum, out_sum, pos, tok_lens, in_deg, out_deg], dim=1)
    feats = torch.cat([feats, role_flags, agent_onehot, agent_norm, tool_flags], dim=1)
    to_final = weights[:, -1].unsqueeze(1) if t > 0 else torch.zeros((t, 1))
    feats = torch.cat([feats, to_final], dim=1)
    feats = torch.cat([feats, sim_to_final.unsqueeze(1), dist_to_final.unsqueeze(1)], dim=1)
    if anomaly.numel() == t:
        feats = torch.cat([feats, anomaly.unsqueeze(1)], dim=1)
    else:
        feats = torch.cat([feats, torch.zeros((t, 1))], dim=1)
    return feats
