# AG Graph Sweep Log

Date: 2025-01-25

## Scope
- Goal: improve AG step-level (when) without hurting who.
- Fixed settings unless noted: model_type=mlp, anomaly_mode=proto, train_epochs=5, train_lr=5e-4, split_ratio=0.5, max_length=512, max_steps=16, max_chars_per_step=400, last_n_layers=2, normalize_by_src_len, depth/top_k/renorm scanned, head_agg scanned.
- Dataset: `/root/autodl-tmp/Attribution_Final/data/Who&When` (Algorithm-Generated)

## Baseline (from summary)
- top_k=5, depth=6, renorm=on, head_agg=mean
- when_acc 0.2063 (13/63), who_acc 0.5238 (33/63)

## Sweep: top_k (depth=6, renorm=on, head_agg=mean)

| top_k | when | who | notes |
|---|---|---|---|
| 2 | 0.1905 (12/63) | 0.5238 (33/63) | worse when |
| 4 | 0.2063 (13/63) | 0.5238 (33/63) | same as baseline |
| 6 | 0.2063 (13/63) | 0.5397 (34/63) | who slight + |
| 8 | 0.2063 (13/63) | 0.5397 (34/63) | neutral vs baseline |
| 12 | 0.2063 (13/63) | 0.5397 (34/63) | who slight + |

## Sweep: depth (top_k=5, renorm=on, head_agg=mean)

| depth | when | who | notes |
|---|---|---|---|
| 2 | 0.1905 (12/63) | 0.5079 (32/63) | worse |
| 3 | 0.1905 (12/63) | 0.5079 (32/63) | worse |
| 4 | 0.2063 (13/63) | 0.5079 (32/63) | who drop |
| 6 | 0.2063 (13/63) | 0.5238 (33/63) | baseline |

## Sweep: renorm (top_k=5, depth=6, head_agg=mean)

| renorm | when | who | notes |
|---|---|---|---|
| on | 0.2063 (13/63) | 0.5238 (33/63) | baseline |
| off | 0.2222 (14/63) | 0.5397 (34/63) | best so far |

## Sweep: head_agg (top_k=5, depth=6, renorm=on)

| head_agg | when | who | notes |
|---|---|---|---|
| mean | 0.2063 (13/63) | 0.5238 (33/63) | baseline |
| max | 0.2222 (14/63) | 0.5238 (33/63) | when improves |

## Check: head_agg max + renorm off
- when_acc 0.2063 (13/63)
- who_acc 0.5238 (33/63)
- note: no improvement when combining max head_agg with renorm off.

## Sweep: renorm_mode=l1 + min_edge_weight (top_k=5, depth=6, head_agg=mean)

| min_edge_weight | when | who | notes |
|---|---|---|---|
| 0.0005 | 0.2063 (13/63) | 0.5079 (32/63) | who drop |
| 0.001 | 0.2063 (13/63) | 0.5397 (34/63) | who slight + |

## Sweep: renorm_mode=l1 + keep_ratio (top_k=5, depth=6)

| keep_ratio | when | who | notes |
|---|---|---|---|
| 0.5 | 0.1746 (11/63) | 0.5397 (34/63) | worse when |

## Sweep: renorm_mode=softmax + temperature (top_k=5, depth=6, min_edge_weight=0.0005)

| temperature | when | who | notes |
|---|---|---|---|
| 0.5 | 0.1587 (10/63) | 0.5714 (36/63) | when drops, who improves |
| 1.0 | 0.1587 (10/63) | 0.5714 (36/63) | same as 0.5 |
| 2.0 | 0.1587 (10/63) | 0.5714 (36/63) | same as 0.5 |

## Sweep: head_select=top_entropy (top_k=5, depth=6, renorm=none)

| head_top_k | when | who | notes |
|---|---|---|---|
| 4 | 0.2222 (14/63) | 0.5397 (34/63) | ties best |
| 8 | 0.2222 (14/63) | 0.5238 (33/63) | who drop |

## Sweep: overlap_weight (top_k=5, depth=6, renorm=none)

| overlap_weight | when | who | notes |
|---|---|---|---|
| 0.2 | 0.2063 (13/63) | 0.5238 (33/63) | neutral |
| 0.5 | 0.1905 (12/63) | 0.5238 (33/63) | worse when |

## Sweep: renorm_mode=none + min_edge_weight

| min_edge_weight | when | who | notes |
|---|---|---|---|
| 0.0001 | 0.2222 (14/63) | 0.5238 (33/63) | same when |
| 0.0002 | 0.2222 (14/63) | 0.5238 (33/63) | same when |
| 0.0005 | 0.1905 (12/63) | 0.5079 (32/63) | worse |
| 0.001 | 0.2540 (16/63) | 0.4921 (31/63) | best when, who drop |

## Sweep: prune_mode (renorm=none, top_k=5, depth=6)

| prune_mode | params | when | who | notes |
|---|---|---|---|---|
| quantile | keep_ratio=0.9 | 0.2063 (13/63) | 0.5079 (32/63) | who drop |
| zscore | zscore_k=1.0 | 0.2063 (13/63) | 0.4603 (29/63) | who drop |
| soft | keep_ratio=0.9 | 0.1746 (11/63) | 0.5238 (33/63) | when drop |

## Failure step strategy (renorm=none, top_k=5, depth=6)

| strategy | when | who | notes |
|---|---|---|---|
| final | 0.2222 (14/63) | 0.5238 (33/63) | baseline |
| max_anomaly | 0.2222 (14/63) | 0.5079 (32/63) | who drop |
| first_error | 0.2540 (16/63) | 0.4762 (30/63) | when up, who drop |

## Who decision mode (renorm=none, min_edge_weight=0.001)

| who_mode | who_hops | when | who | notes |
|---|---|---|---|---|
| global | - | 0.2540 (16/63) | 0.4921 (31/63) | baseline for this setting |
| neighborhood | 2 | 0.2540 (16/63) | 0.3492 (22/63) | who drop |

## Two-stage who (agent_scores / agent_sim) with logsumexp

| step_pred_mode | when | who | notes |
|---|---|---|---|
| agent_scores | 0.0317 (2/63) | 0.4921 (31/63) | when collapses |
| agent_sim | 0.1270 (8/63) | 0.4921 (31/63) | when drop |

## SequenceScorer (seq) + embed_extra=scores_pos

| model_type | when | who | notes |
|---|---|---|---|
| seq | 0.1587 (10/63) | 0.1429 (9/63) | unstable / worse |

## Multi-sink + Counterfactual (Δ)

| config | when | who | notes |
|---|---|---|---|
| static only (multi-sink) | 0.2222 (14/63) | 0.4921 (31/63) | cf disabled |
| + Δ(v) (mask, cf_top_k=3, eta=1.0) | 0.2698 (17/63) | 0.4921 (31/63) | when improves |

## Path-based who (with multi-sink + Δ)

| who_mode | when | who | notes |
|---|---|---|---|
| path_based | 0.2698 (17/63) | 0.3175 (20/63) | who drops |

## Δ-aligned who (multi-sink + Δ)

| config | when | who | notes |
|---|---|---|---|
| renorm=none, min_edge_weight=None | 0.2222 (14/63) | 0.3333 (21/63) | who drop; when back to baseline |
| renorm=none, min_edge_weight=0.001 | 0.2063 (13/63) | 0.3651 (23/63) | who drop; when degrades |

## Sink-subgraph who (multi-sink + Δ)

| config | when | who | notes |
|---|---|---|---|
| subgraph_weight=delta, top_n=8, top_m=3 | 0.2063 (13/63) | 0.3810 (24/63) | who drop; when degrades |

## Pseudo-normal (prefix) + NLL anomaly

| config | when | who | notes |
|---|---|---|---|
| proto_nll + zscore, prefix_ratio=0.5, sinks=final+anomaly, lr=5e-5 | 0.0952 (6/63) | 0.1270 (8/63) | severe drop |
| proto_nll + zscore, prefix_ratio=0.5, sinks=final, lr=5e-4 | 0.2381 (15/63) | 0.4444 (28/63) | when < 0.25 |
| proto_nll + zscore, prefix_ratio=0.5, sinks=final+anomaly, lr=5e-4 | 0.2381 (15/63) | 0.4921 (31/63) | who recovers, when flat |
| proto + zscore, prefix_ratio=0.5, sinks=final+anomaly | 0.2381 (15/63) | 0.5238 (33/63) | no gain |
| proto + none, prefix_ratio=0.5, sinks=final+anomaly | 0.2222 (14/63) | 0.5079 (32/63) | worse |

## Proto bank + NLL gap (context-gap)

| config | when | who | notes |
|---|---|---|---|
| proto bank + zscore, sinks=final+anomaly | 0.1587 (10/63) | 0.5079 (32/63) | worse when |
| proto_nll bank + zscore, gap=2, sinks=final+anomaly | 0.2063 (13/63) | 0.4921 (31/63) | flat |
| proto_nll bank + zscore, gap=2, sinks=final | 0.1905 (12/63) | 0.4921 (31/63) | worse |

## Node-agg who (top-K nodes)

| config | when | who | notes |
|---|---|---|---|
| node_agg + influence_anomaly, top_k=5 | 0.2222 (14/63) | 0.3175 (20/63) | who drop |
| node_agg + influence, top_k=5 | 0.2222 (14/63) | 0.3175 (20/63) | who drop |

## Light anomaly + agent aggregation (current)

| config | when | who | notes |
|---|---|---|---|
| proto_delta (anomaly), agent_agg=sum | 0.1746 (11/63) | 0.5397 (34/63) | when drop |
| proto (anomaly), agent_agg=topk (k=3) | 0.2222 (14/63) | 0.5238 (33/63) | no gain |
| proto_delta (anomaly), agent_agg=topk (k=3) | 0.1746 (11/63) | 0.5397 (34/63) | when drop |

## Prefix-only training + sampling (AG)

| config | when | who | notes |
|---|---|---|---|
| train_prefix_only + history_sample=head | 0.0000 (0/63) | 0.2540 (16/63) | collapse |
| train_prefix_only + history_sample=uniform | 0.0000 (0/63) | 0.2540 (16/63) | collapse |
| history_sample=head (no prefix) | 0.2222 (14/63) | 0.5238 (33/63) | same as baseline |
| history_sample=uniform (no prefix) | 0.2222 (14/63) | 0.5238 (33/63) | same as baseline |

## Diagnostics
- DOT + text reports generated under `experiments/diagnostics/` (2 failed AG samples) with top‑N incoming edges and pruned edge lists.
- 最新报告包含 delta‑aligned 与 subgraph 的 AgentScore 分解（txt/json）。
- delta‑aligned 诊断已输出到 `experiments/diagnostics/delta_aligned/`。

## Diagnostic Conclusions (from 2 failed cases)
- Case A (fail_0): evidence_path=[0,9], kept incoming edges are all 0; pruned edges have weights ~8e-4–1e-3. **裁剪型**为主：关键边存在于未裁剪集合，但被 `min_edge_weight=0.001` 截断；同时出现 **缺失/稀释** 现象（保留边权为 0）。
- Case B (fail_1): evidence_path=[0,7], kept incoming edges为 0，pruned edges 仍有 ~5e-4–8e-4 的非零权。结论同上：**裁剪型为主，平滑/缺失次要**。

Implication: 在 AG 上固定阈值会切掉“相对重要但绝对值偏小”的依赖边；应优先采用自适应裁剪或更柔性的阈值策略。

## Failure Case Summary (placeholder)
- Visualization not implemented yet. Plan: add DOT export for a failed sample and log the max-weight path.
