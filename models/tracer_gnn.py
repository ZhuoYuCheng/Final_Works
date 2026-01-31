import torch
import torch.nn as nn


class StepScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


class GraphScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() == 0:
            return self.fc2(torch.relu(self.fc1(feats))).squeeze(-1)
        agg = torch.matmul(weights, feats)
        h = torch.relu(self.fc1(feats + agg))
        return self.fc2(h).squeeze(-1)
