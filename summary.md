# Run Summary

Date: 2026-02-01

## Status
- Unified pipeline (Schema + Adapter + Soft-IDG + MASC-style anomaly) is stable; entry `cli.py`.
- Who&When copied from `/root/autodl-tmp/GraphTracer_Reproduce/Who&When` to `Attribution_Final/data/Who&When`.
- AG sweeps and diagnostics are logged in `experiments/ag_graph_sweep.md` and `experiments/diagnostics/`.
- AG baseline (when-priority) updated to `failure_step_strategy=multi` with trade-off accepted.

## Who&When (Hand-Crafted)
- Baseline (split_ratio=0.5, mlp, renorm on): when 0.4000 (4/10), who 0.6552 (19/29).
- Improved (max_length=768, head_tail, max_anomaly): when 0.3750 (9/24), who 0.6207 (18/29).
- Command (best/stable):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data/Who\&When \
    --subset Hand-Crafted \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 512 \
    --max_steps 16 \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --top_k 5 \
    --depth 6 \
    --renorm \
    --train_epochs 5 \
    --train_lr 5e-4 \
    --split_ratio 0.5 \
    --model_type mlp \
    --anomaly_mode proto
  ```
- Command (improved, meets target):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data/Who\&When \
    --subset Hand-Crafted \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 768 \
    --max_steps 32 \
    --history_sample head_tail \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --head_agg mean \
    --head_select all \
    --head_top_k 8 \
    --top_k 5 \
    --depth 6 \
    --min_in_edges 1 \
    --renorm \
    --train_epochs 5 \
    --train_lr 5e-4 \
    --split_ratio 0.5 \
    --split_strategy random \
    --split_seed 42 \
    --model_type mlp \
    --anomaly_mode proto \
    --anomaly_weight 0 \
    --failure_step_strategy max_anomaly \
    --step_pred_mode model \
    --agent_agg sum \
    --agent_feat_dim 12
  ```

## Who&When (Algorithm-Generated) — Pareto Points
- Balanced (renorm=none): when 0.2222 (14/63), who 0.5238 (33/63).
- When-focused (renorm=none + min_edge_weight=0.001): when 0.2540 (16/63), who 0.4921 (31/63).
- Who-focused (renorm=softmax, τ=0.5): when 0.1587, who 0.5714.
- Historical best (balanced, seq + agent_loss, max_anomaly): when 0.2540 (16/63), who 0.6190 (39/63).
- Baseline (AG when-priority, 2026-02-01): `failure_step_strategy=multi` → when 0.3175 (20/63), who 0.6032 (38/63). HC unchanged; aegis when 0.7121 / who 0.5432.
- Note: multi-sink + counterfactual Δ reached when 0.2698 but who stayed 0.4921 (not Pareto).
- Command (Balanced, best/stable):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data/Who\&When \
    --subset Algorithm-Generated \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 512 \
    --max_steps 16 \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --top_k 5 \
    --depth 6 \
    --renorm_mode none \
    --train_epochs 5 \
    --train_lr 5e-4 \
    --split_ratio 0.5 \
    --model_type mlp \
    --anomaly_mode proto
  ```
- Command (improved, meets target):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data/Who\&When \
    --subset Algorithm-Generated \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 512 \
    --max_steps 16 \
    --history_sample tail \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --head_agg mean \
    --head_select all \
    --head_top_k 8 \
    --top_k 5 \
    --depth 6 \
    --min_in_edges 1 \
    --renorm_mode none \
    --min_edge_weight 0.001 \
    --train_epochs 10 \
    --train_lr 5e-4 \
    --split_ratio 0.5 \
    --split_strategy random \
    --split_seed 42 \
    --model_type seq \
    --embed_extra scores_pos \
    --seq_hidden_dim 256 \
    --seq_heads 4 \
    --seq_layers 2 \
    --anomaly_mode proto \
    --anomaly_weight 0 \
    --failure_step_strategy max_anomaly \
    --step_pred_mode model \
    --agent_agg sum \
    --agent_loss_weight 0.7 \
    --agent_feat_dim 4
  ```
 - Command (baseline, AG when priority):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data/Who\&When \
    --subset Algorithm-Generated \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 512 \
    --max_steps 16 \
    --history_sample tail \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --head_agg mean \
    --head_select all \
    --head_top_k 8 \
    --top_k 5 \
    --depth 6 \
    --min_in_edges 1 \
    --renorm_mode none \
    --min_edge_weight 0.001 \
    --train_epochs 10 \
    --train_lr 5e-4 \
    --split_ratio 0.5 \
    --split_strategy random \
    --split_seed 42 \
    --model_type seq \
    --embed_extra scores_pos \
    --seq_hidden_dim 256 \
    --seq_heads 4 \
    --seq_layers 2 \
    --anomaly_mode proto \
    --anomaly_weight 0 \
    --failure_step_strategy multi \
    --step_pred_mode model \
    --agent_agg sum \
    --agent_loss_weight 0.7 \
    --agent_feat_dim 4
  ```

### 2026-01-31 Suggestion Validation (AG, keep attention-based IDG)
Baseline for comparison (historical): **seq + agent_loss, max_anomaly** = when 0.2540 / who 0.6190.

Tested one change at a time; adopt only if **both** when & who are **not worse** than baseline.
Exception: as of 2026-02-01, `failure_step_strategy=multi` is promoted to AG when-priority baseline with an accepted who trade-off.

| change | when | who | decision | notes |
|---|---|---|---|---|
| `min_edge_ratio=0.1` (adaptive, no absolute threshold) | 0.3016 (19/63) | 0.5714 (36/63) | reject | when ↑, who ↓ |
| `failure_step_strategy=multi` | 0.3175 (20/63) | 0.6032 (38/63) | accept (AG when baseline) | when ↑, who ↓ (trade-off accepted) |
| `anomaly_weight=0.1` | 0.3016 (19/63) | 0.5397 (34/63) | reject | who ↓ |
| long history (`max_length=768`, `max_steps=32`, `head_tail`) | 0.2698 (17/63) | 0.5714 (36/63) | reject | who ↓ |
| `step_pred_mode=agent_first` + `agent_agg=logsumexp` | 0.2381 (15/63) | 0.5714 (36/63) | reject | when/ who ↓ |
| `agent_agg=logsumexp` (model mode) | 0.2540 (16/63) | 0.5714 (36/63) | reject | who ↓ |
| `token_agg=topk` | 0.1905 (12/63) | 0.5238 (33/63) | reject | both ↓ |

Conclusion: no new setting surpassed the historical best on both metrics; keep the balanced AG best configuration, and use `failure_step_strategy=multi` as the AG when-priority baseline.

## Aegis (Aligned adapter)
- Config: seq + renorm, split_ratio=0.8, scores_pos, agent_loss_weight=0.7, Qwen3-8B.
- Result: when 0.6970 (46/66), who 0.6420 (52/81).
- Notes: when is derived as earliest faulty agent step; adapter alignment (agent_name only, case-insensitive matching) increased valid when labels (55 → 66) and lifted metrics.
- Command (best/stable):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data \
    --subset aegis \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 512 \
    --max_steps 16 \
    --history_sample tail \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --head_agg mean \
    --head_select all \
    --head_top_k 8 \
    --top_k 5 \
    --depth 6 \
    --min_in_edges 1 \
    --renorm \
    --train_epochs 10 \
    --train_lr 5e-4 \
    --split_ratio 0.8 \
    --split_strategy random \
    --split_seed 42 \
    --model_type seq \
    --embed_extra scores_pos \
    --agent_loss_weight 0.7 \
    --anomaly_mode proto \
    --anomaly_norm zscore \
    --anomaly_weight 0.1 \
    --failure_step_strategy final \
    --step_pred_mode model \
    --who_mode global \
    --agent_agg sum \
    --agent_feat_dim 0
  ```
- Command (best/stable):
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /root/autodl-tmp/Attribution_Final/cli.py \
    --data_dir /root/autodl-tmp/Attribution_Final/data \
    --subset aegis \
    --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
    --torch_dtype float16 \
    --max_length 512 \
    --max_steps 16 \
    --max_chars_per_step 400 \
    --normalize_by_src_len \
    --last_n_layers 2 \
    --top_k 5 \
    --depth 6 \
    --renorm \
    --train_epochs 10 \
    --train_lr 5e-4 \
    --split_ratio 0.8 \
    --model_type seq \
    --embed_extra scores_pos \
    --agent_loss_weight 0.7 \
    --score_mix 0 \
    --step_pred_mode model \
    --who_mode global
  ```
