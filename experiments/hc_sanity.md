# HC Long-History Sanity Check

Date: 2025-01-25

## Goal
Verify long-history robustness (head_tail sampling + early bias) without focusing on HC optimization.

## Command
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python /root/autodl-tmp/Attribution_Final/cli.py \
  --data_dir /root/autodl-tmp/Attribution_Final/data/Who\&When \
  --subset Hand-Crafted \
  --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
  --torch_dtype float16 \
  --max_length 768 \
  --max_steps 28 \
  --history_sample head_tail \
  --max_chars_per_step 60 \
  --normalize_by_src_len \
  --last_n_layers 2 \
  --top_k 5 \
  --depth 6 \
  --renorm \
  --train_epochs 5 \
  --train_lr 5e-4 \
  --split_ratio 0.5 \
  --model_type mlp \
  --anomaly_mode proto \
  --early_bias_weight 2.0 \
  --early_bias_power 1.0
```

## Result
- when_acc 0.2273 (5/22)
- who_acc 0.2759 (8/29)

## Notes
- when/ who 均低于基线，仅作为鲁棒性 sanity check；不纳入 summary。

---

## HC Check: delta-aligned who (multi-sink + Δ)

## Command
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
  --renorm_mode none \
  --min_edge_weight 0.001 \
  --train_epochs 5 \
  --train_lr 5e-4 \
  --split_ratio 0.5 \
  --model_type mlp \
  --anomaly_mode proto \
  --failure_sinks_mode final+first_error+anomaly \
  --sink_top_m 3 \
  --cf_top_k 3 \
  --intervention_mode mask \
  --cf_eta 1.0 \
  --who_mode delta_aligned \
  --delta_top_k 3
```

## Result
- when_acc 0.3000 (3/10)
- who_acc 0.2069 (6/29)

## Notes
- HC 上 who 明显下降；该配置只用于验证机制，不纳入 summary。
