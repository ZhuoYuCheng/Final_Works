from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


def serialize_steps(
    steps, max_chars_per_step: Optional[int] = None
) -> Tuple[str, List[Tuple[int, int]]]:
    blocks = []
    for idx, step in enumerate(steps):
        role = step.role or ""
        name = step.name or ""
        content = step.output or ""
        if max_chars_per_step is not None and len(content) > max_chars_per_step:
            content = content[:max_chars_per_step]
        header = f"Step {idx} | Role {role}"
        if name:
            header = f"{header} | Name {name}"
        block = f"[{header}]\n{content}".strip()
        blocks.append(block)

    spans = []
    parts = []
    cursor = 0
    for block in blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(block)
        cursor += len(block)
        end = cursor
        spans.append((start, end))

    return "".join(parts), spans


class SLMWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 2048,
        last_n_layers: int = 4,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        head_agg: str = "mean",
        head_select: str = "all",
        head_top_k: int = 8,
        layer_weighting: str = "uniform",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.last_n_layers = last_n_layers
        self.head_agg = head_agg
        self.head_select = head_select
        self.head_top_k = head_top_k
        self.layer_weighting = layer_weighting
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        self.model.to(self.device)
        self.model.eval()

    def _layer_weights(self, n: int) -> torch.Tensor:
        if self.layer_weighting == "late":
            weights = torch.linspace(1.0, 2.0, steps=n)
        elif self.layer_weighting == "early":
            weights = torch.linspace(2.0, 1.0, steps=n)
        else:
            weights = torch.ones(n)
        return weights / weights.sum()

    def _select_heads(self, attn: torch.Tensor) -> torch.Tensor:
        if self.head_select == "all":
            return attn
        # attn: H x S x S
        h, s, _ = attn.size()
        if h == 0:
            return attn
        probs = attn.clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=-1)  # H
        k = min(self.head_top_k, h)
        _, idx = torch.topk(-entropy, k=k)
        return attn[idx]

    @torch.no_grad()
    def encode_text(self, text: str) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_attentions=True, output_hidden_states=False)
        attn = outputs.attentions
        if attn is None:
            raise RuntimeError("Attentions are None. Ensure eager attention is enabled.")
        hidden = outputs.last_hidden_state.squeeze(0).cpu()
        total_layers = len(attn)
        start = max(0, total_layers - self.last_n_layers)
        selected = attn[start:]
        stacked = torch.stack(selected, dim=0)
        weights = self._layer_weights(stacked.size(0)).to(stacked.device)
        layer_mean = (stacked * weights[:, None, None, None, None]).sum(dim=0)
        if layer_mean.dim() == 4 and layer_mean.size(0) == 1:
            layer_mean = layer_mean.squeeze(0)
        if self.head_agg == "max":
            selected_heads = self._select_heads(layer_mean)
            attn_mean = selected_heads.max(dim=0).values
        else:
            selected_heads = self._select_heads(layer_mean)
            attn_mean = selected_heads.mean(dim=0)
        return attn_mean.cpu(), offsets, hidden
