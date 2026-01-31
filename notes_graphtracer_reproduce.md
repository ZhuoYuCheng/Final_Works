# GraphTracer_Reproduce Reference Notes (for Attribution_Final)

Date: 2025-01-25

## 1) IDG 构建与边处理策略

**代码入口**
- `GraphTracer_Reproduce/core_code/attn_graph.py`
  - `build_soft_dependency_graph(...)`
  - `sparsify_weights(...)`
  - `backtrace_scores(...)`

**边权定义/聚合**
- Token->step 的聚合：
  - 先取每个目标 step j 的 token attention 子矩阵 `attn_matrix[tok_j]`。
  - 对每个来源 step i 的 token 集合 `tok_i` 汇总 attention mass：
    - `token_agg` 可为 `mean / max / topk`（默认 mean）。
  - 可选 `normalize_by_src_len`：`w = w / len(tok_i)`。
- **Head 聚合**：`head_agg` 支持 `mean / max`，默认 mean。
- **Layer 聚合**：仅对最后 `last_n_layers` 做均值。

**边裁剪/归一**
- `sparsify_weights` 采用 **per-node top_k**（对每个 j 的入边）。
- 可选 `threshold` 绝对阈值裁剪。
- `renorm` 默认对每个 j 的入边做 L1 归一（per-node）。
- 没有温度 softmax / 自适应分位数裁剪 / z-score 裁剪。

**对比当前 Attribution_Final**
- GraphTracer_Reproduce 是 **“per-node top_k + L1 renorm”** 的简化策略。
- 未提供保护“相对重要但绝对值小”的边的机制（仅 top_k / threshold）。
- 我们当前扩展了 `min_edge_weight / keep_ratio / renorm_mode / temperature` 等精细裁剪，但仍需更温和的 **自适应** 策略才能避免 who 下滑。

**可迁移/复刻的机制**
- 保留 per-node top_k 作为硬边数上限（避免过密）。
- `token_agg=topk` 与 `head_agg=max` 的可切换性是可复用对比项。

---

## 2) Failure step / failure node 定义

**默认策略**
- `backtrace_scores` 在 `t_f = t-1` 处起始（即 final step 作为 failure node）。
- 没有 “first_error / max_anomaly” 等显式替代策略。

**对比当前 Attribution_Final**
- 我们已引入 `failure_step_strategy`（final / max_anomaly / first_error），并实证：
  - `first_error` 能提升 when，但明显拉低 who。

**可迁移/复刻的机制**
- GraphTracer_Reproduce 没有 failure node 选择机制，可作为 baseline 解释（final-step 假设）。
- 若要追求 AG 的 when，应继续把 “failure node 的可变选择” 作为机制改造方向，而非单纯图参数调优。

---

## 3) 根因与 who 输出方式

**Step → who 的映射**
- 默认：`pred_when = argmax(scores)`，`pred_who = agent(pred_when)`。
- 可选：`step_pred_mode` 支持 `agent_scores / agent_sim / agent_direct`，先预测 agent 再在该 agent 的 step 中找最大分数。
- `aggregate_agent_logits` 支持 `agent_agg=logsumexp` 或 `max` 做 agent 聚合。
- 训练时可加入 `agent_loss_weight` 辅助 who。

**对比当前 Attribution_Final**
- 当前默认仍是 argmax step → agent。
- 我们新增了 `who_mode=neighborhood`（邻域投票）但在 AG 上未改善 who。

**可迁移/复刻的机制**
- GraphTracer_Reproduce 的 `agent_scores / agent_sim / agent_direct` 逻辑可直接迁移，属于 “先定 agent 再选 step” 的 two‑stage 轻量方法。
- `agent_agg=logsumexp` 比 `max` 更稳健（对噪声鲁棒）。

---

## 4) 可解释诊断与日志

**GraphTracer_Reproduce**
- 未提供 DOT/报告输出；仅在 README 记录参数与结果。
- 可解释性主要依赖 `scores` 与 `backtrace` 的中间量，但未落盘。

**Attribution_Final 当前**
- 已补齐：`experiments/diagnostics/` 下的 DOT + 文本，包含 failure step、evidence_path、top‑N 入边与裁剪边摘要。

**可迁移的最小诊断实现**
- 保持 per‑node top‑N incoming edges 与 pruned edges 的对照展示（已实现）。

---

## 5) 默认超参与关键差异

**GraphTracer_Reproduce（README best runs）**
- HC：`top_k=5`, `depth=6`, `renorm=on`, `model_type=gnn`, `score_mix=5`, `step_pred_mode=agent_sim`
- AG：`top_k=5`, `depth=6`, `renorm=on`, `model_type=seq`, `score_mix=0`, `step_pred_mode=model`

**Attribution_Final（当前）**
- 基线：`top_k=5`, `depth=6`, `renorm=on`, `model_type=mlp`
- AG When‑focused：`renorm=none + min_edge_weight=0.001`

**关键差异点**
- GraphTracer_Reproduce 在 AG 使用 **SequenceScorer**（`model_type=seq`）与 `embed_extra`（scores+pos），而我们目前主线仍是 MLP。
- GraphTracer_Reproduce 的 who 预测可通过 `agent_scores/agent_sim` 两阶段策略，而我们主线多为全局 argmax step。

---

## 6) 迁移/复刻的 2–3 个最可能带来突破的机制

1) **Two‑stage agent‑first 规则（GraphTracer 的 `agent_scores/agent_sim`）**
   - 先预测 agent，再在该 agent 的 step 中选 when，可能同时提升 who 并保持 when。
2) **SequenceScorer (seq) + embed_extra=scores_pos**
   - AG 上的 best setting 使用 seq，意味着“序列模型 + 位置/score 特征”对 noisy AG 更稳。
3) **保持 per‑node top_k + soft threshold**
   - GraphTracer 的 per‑node top_k 已验证可行；结合我们诊断结论，应在 per‑node 内做软阈值，而非全局 min_edge_weight。

---

## 结论（指导下一步）
- GraphTracer_Reproduce 的默认策略偏向 **per-node top_k + L1 renorm + final step backtrace**，没有对“绝对小但相对重要”边的保护。
- AG 的 trade‑off 可能更应从 **step→who 的两阶段规则** 与 **SequenceScorer** 入手，而非继续堆砌图参数。
- 下一阶段建议优先复刻：`agent_scores/agent_sim` 与 `model_type=seq` 的组合，再对自适应裁剪做小规模微调。
