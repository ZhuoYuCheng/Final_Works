import torch
import torch.nn as nn


class SequenceScorer(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        h = self.encoder(h.unsqueeze(0)).squeeze(0)
        return self.out(h).squeeze(-1)
