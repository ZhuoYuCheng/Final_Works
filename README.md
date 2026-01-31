# README — White-Box Fault Attribution for Multi-Agent Systems (Who & When)

本项目实现一个面向多智能体系统（MAS）的**高效故障归因**框架：在多轮交互轨迹中自动定位“**哪个智能体（Who）在第几步（When）**引入错误”，并给出**错误传播路径**与可视化解释。框架强调使用开源小模型（SLM，如 Qwen 系列）的**白盒信号**（注意力/隐藏表征等）进行诊断，避免昂贵的闭源 LLM 全日志裁判。 

核心思想融合两条路线：

* **MASC**：将 step-level 误差检测视为历史条件的无监督异常检测（next-execution reconstruction + prototype prior）以产生异常分数 `s(t)`。 
* **GraphTracer**：通过信息依赖图（IDG）回溯依赖链，避免将“症状”误判为“根因”。本项目用**注意力统计构造 Soft-IDG**替代复杂解析建图。 

---

## 1. 目标与范围

### 1.1 输入

* 任意 MAS 执行轨迹（Who&When 数据集或未来新数据集），包含：

  * 任务 query
  * 多步交互：每步的 `agent/role`、输出文本、可选工具调用与结果、可选环境反馈
  * 训练/评测时可含 `who/when` 标签

### 1.2 输出

* `pred_when`: 根因步号（step index）
* `pred_who`: 根因智能体（agent id/role）
* `evidence_path`: 从根因到失败显现步的依赖路径（节点序列 + 边权/贡献）
* 可选：`report.json` + `graph.dot`/`graph.html` 可视化产物

---

## 2. 为什么必须做“统一内部数据 Schema + Adapter 转换层”

Who&When 的原始 JSON 结构与本项目内部处理所需结构可能不一致，且未来将接入新数据集。工程上应采取**“适配层一次性消化差异，算法层只依赖统一 Schema”**的模式：

* 避免数据集差异渗透到 detector / graph / tracer / eval 各模块
* 保障 Soft-IDG 建图、token span 对齐、路径回溯等对字段语义的稳定假设
* 新数据集接入只需新增 `Adapter`，无需修改算法代码

结论：**必须先转换成统一内部 Schema，然后全链路只处理统一 Schema**。

---

## 3. 系统架构

```
Raw Dataset JSON(s)
  -> Adapter Layer (per-dataset convert)
    -> Unified Trajectory Schema (canonical)
      -> White-Box Feature Extractor (SLM attention / embeddings / spans)
        -> Unsupervised Step Detector (MASC-style) => anomaly score s(t)
          -> Soft-IDG Builder (attention-based dependency graph)
            -> Root Cause Tracer (graph backtrace + scoring; optional GNN)
              -> Evaluator + Visualizer
```

---

## 4. 目录结构（建议）

```

数据布局（当前实现）：

- Who&When：`/root/autodl-tmp/Attribution_Final/data/Who&When`
  - 由 `/root/autodl-tmp/GraphTracer_Reproduce/Who&When` 复制
Attribution_Final/
  data/
    schema.py                 # UnifiedTrajectory/Step/Label dataclasses
    validate.py               # schema validation + fallback policies
    adapters/
      base.py                 # BaseAdapter interface
      who_when.py             # WhoWhenAdapter (raw->unified)
  datasets/
    registry.py               # dataset registry + loaders
  models/
    slm_wrapper.py            # HF Transformers wrapper: attn/emb/spans
    masc_detector.py          # prototype anomaly score s(t)
    soft_idg.py               # build Soft-IDG from attention
    tracer_rule.py            # rule tracer (backtrace + anomaly mix)
    tracer_gnn.py             # optional supervised GNN tracer
    feature_builder.py        # step-level feature construction
  eval/
    metrics.py                # who_acc / when_acc
    ablation.py               # component ablations (placeholder)
  viz/
    export_graphviz.py        # DOT export (placeholder)
    render_html.py            # interactive HTML (placeholder)
  cli.py                      # train/eval entrypoint
  summary.md                  # run log
  README.md
```

---

## 5. 统一内部数据 Schema（Canonical）

### 5.1 设计原则

* **最小核心字段稳定**：算法层只依赖这些字段
* **可扩展**：未来新数据集字段放入 `meta` / `raw`
* **可降级**：缺字段时有明确 fallback

### 5.2 Canonical JSON（示例）

```json
{
  "id": "trace_001",
  "query": "....",
  "agents": [
    {"agent_id": 0, "role": "planner"},
    {"agent_id": 1, "role": "executor"}
  ],
  "steps": [
    {
      "t": 1,
      "agent_id": 0,
      "role": "planner",
      "input_ctx_ref": [0],
      "output": "....",
      "events": [
        {"type": "tool_call", "name": "search", "args": {"q": "..." }},
        {"type": "tool_result", "name": "search", "result": "..."}
      ],
      "raw": {}
    }
  ],
  "labels": {"when": 5, "who": 1},
  "meta": {"dataset": "who_when", "subset": "handcrafted", "gt_available": true}
}
```

#### 字段说明

* `steps[t].input_ctx_ref`：该步模型实际可见上下文的 step 索引列表（若原数据无该字段，按 fallback 生成；见 6.2）
* `events`：统一表达 tool / environment / feedback 等信息，避免不同数据集结构差异
* `raw`：保留原始数据片段，便于调试和回溯

---

## 6. Adapter 设计（关键工程层）

### 6.1 BaseAdapter 接口

每个数据集实现：

* `can_handle(raw_obj) -> bool`
* `convert(raw_obj) -> UnifiedTrajectory`
* `validate(unified_obj) -> ValidationReport`（可复用统一校验器）

### 6.2 必须实现的 fallback 策略

为了兼容未来数据集缺字段情况：

1. **缺失 `input_ctx_ref`**
   默认策略之一（写入 config）：

* `all_history`: `input_ctx_ref = [0..t-1]`
* `sliding_window(K)`: 最近 K 步
* `role_filtered`: 仅保留特定 role（如 planner/critic）

2. **缺失工具结构**
   将原数据中任何 observation / tool log / feedback 统一映射到 `events[]`（尽量保留原始信息）

3. **缺失 who/when 标签**
   仍可运行无监督归因（输出 pred），评测时跳过 supervised 指标或仅做定性分析

---

## 7. White-Box 特征抽取（SLM Wrapper）

### 7.1 必须输出

* `step_embedding[t]`: 每步 embedding（建议 last hidden EOS 或 mean pooling）
* `attn[t]`: 注意力张量（建议聚合到 “输出token -> 输入token” 的可用统计）
* `token_spans[t]`: 该 step 在拼接输入序列中的 token span（Soft-IDG 依赖这个）

### 7.2 强建议：缓存

* embedding/attn 计算成本高，必须落盘缓存（按 `trace_id + step` key）
* 便于重复建图、重复 tracer 调参、可视化复用

---

## 8. 无监督异常检测（MASC-style Step Detector）

目标：对每一步输出异常分数 `s(t)`，形成疑点热力图，用于召回候选根因。 

### 8.1 关键公式（可实现的简化版）

* next-execution reconstruction：预测当前步 embedding `x̂_t`
* prototype prior：维护正常行为原型 `p`
* anomaly score：

  * `s(t) = α * ||x̂_t - x_t||^2 + β * (1 - cos(x̂_t, p))` 

输出：

* `anomaly_scores[t]`
* `topK_anomaly_steps`

说明：本项目重点是“归因”，不强制实现论文中 correction agent，但 detector 的分数用于候选筛选与可视化非常关键。 

---

## 9. Soft-IDG（注意力驱动的信息依赖图）

GraphTracer 认为 temporal trace 无法表达跨步依赖，必须显式建图（IDG）。本项目用注意力近似依赖关系构造 **Soft-IDG**。 

### 9.1 节点与边

* 节点：step 粒度节点 `v_t`
* 边：`i -> j`（i<j）表示 j 使用了 i 的信息，边权 `w(i,j)`

### 9.2 边权计算（核心可复现）

对步 `j`：

1. 得到输出 token 集 `T_out(j)`
2. 对历史步 `i` 的 token span `T_hist(i)`，计算聚合注意力：

   * `w(i,j) = mean_{tok∈T_out(j)} sum_{h∈T_hist(i)} Attn(tok, h)`
3. 每个节点只保留 top-M 入边（稀疏化，减少噪声）

### 9.3 显式引用融合（可选但强烈建议）

如果通过 prompt 约束 agent 输出引用（例如 `CITED_UIDS=[...]`），则：

* 对引用到的步强制建边并提升权重
* 注意力边作为补充与回退通道
  这与 GraphTracer 的“显式依赖抽取”方向一致，但更轻量。 

---

## 10. Root Cause Tracer（根因定位）

### 10.1 默认（无需训练）两阶段策略

**Stage A：候选召回**

* `C = topK steps by s(t)`（来自 detector）
* 加入失败显现步（通常 final step 或包含负反馈/错误信号的 step）
* 向图上游扩展若干层祖先

**Stage B：图上排名**

* 从失败节点 `v_f` 反向传播影响力（反向 random-walk / personalized PageRank / DP）
* 综合得分：

  * `Score(v) = λ * s(v) + (1-λ) * Influence(v)`
* 预测：

  * `pred_when = argmax Score(v)`
  * `pred_who = agent_id(pred_when)`
* 路径：

  * 输出贡献最大的 `v_root -> ... -> v_f` 路径作为 `evidence_path`

该设计契合 GraphTracer 通过依赖结构区分“症状 vs 根因”的动机。 

### 10.2 可选：监督式 GNN Tracer

在 Soft-IDG 上训练 GAT/GraphSAGE：

* 输入：node features（embedding、s(t)、agent embedding、位置编码等）+ edge weight
* 输出：每节点 root cause 概率
  训练标签来自 Who&When 的 `when`（映射到节点）。用于提升精度与研究深度。

---

## 11. 评测与消融

### 11.1 指标

* `when_acc`: 根因步准确率
* `who_acc`: 根因智能体准确率
* `top-k when_acc`
* （可选）路径质量：与 GT 路径的编辑距离/覆盖率（若有标注或可构造）

### 11.2 评测设置

参考 Who&When 与 MASC/GraphTracer 常见设置：

* `w/ GT` vs `w/o GT`（是否提供 ground truth answer）  
* handcrafted vs automated 子集（若数据集中存在） 

### 11.3 必做消融

* w/o Soft-IDG（只靠异常分数或只靠时间回溯）
* w/o prototype（early steps 性能通常下降） 
* w/o reconstruction（整体检测能力下降） 
* attention-only vs citation-enhanced（验证显式依赖增益） 

---

## 12. 可视化（Debugging UX）

输出两类产物：

1. `graph.dot` / `graph.png`（Graphviz）

* 节点：`t, agent, s(t), Score(t)`
* 边：`w(i,j)`
* 高亮 `evidence_path`

2. `graph.html`（交互式）

* 点击节点显示 step 文本摘要、top 入边、异常分数构成等

---

## 13. CLI（当前实现）

当前版本提供一个统一入口 `cli.py`，完成训练与评测。下面给出**推荐起点**（适合 HC 的长轨迹场景）：

```bash
python cli.py \
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

### 13.1 Best-known configs (current)

Hand-Crafted (current best, meets target):
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
Result: when 0.3750 (9/24), who 0.6207 (18/29)

Algorithm-Generated (current best, meets target):
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
  --train_epochs 5 \
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
Result: when 0.2540 (16/63), who 0.6190 (39/63)

支持的关键参数：

* `--history_sample`: `tail|head|head_tail|uniform|cover_error`，用于长轨迹采样（`cover_error` 尽量保留错误步）
* `--early_bias_weight/--early_bias_power`: 抵消注意力的 recency 偏置
* `--model_type`: `mlp|gnn|seq`
* `--embed_extra`: `none|scores|scores_pos`（seq 使用）
* `--seq_hidden_dim`/`--seq_heads`/`--seq_layers`: seq 模型超参
* `--anomaly_mode`: `proto|proto_delta|proto_prefix|proto_bank|none`（异常分数）
* `--anomaly_norm`: `none|zscore|minmax`（异常分数归一化）
* `--proto_mode`: `global|prefix|bank`（原型构造方式）
* `--proto_bank_size`: 原型 bank 数量（`proto_bank` 时生效）
* `--anomaly_weight`: 异常分数与依赖回溯的融合权重
* `--score_mix`: 规则分数与模型分数的线性融合
* `--head_agg`: 注意力头聚合方式（`mean`/`max`）
* `--head_select`: 注意力头选择（`all`/`top_entropy`）
* `--head_top_k`: 选取的头数量（配合 `head_select`）
* `--layer_weighting`: 层权重（`uniform`/`late`/`early`）
* `--renorm_mode`: 边权归一化方式（`none`/`l1`/`softmax`）
* `--min_edge_weight`: 边权阈值（裁剪弱边）
* `--keep_ratio`: 按入边权分位数裁剪
* `--prune_mode`: 裁剪策略（`fixed`/`quantile`/`zscore`/`soft`）
* `--zscore_k`: z-score 裁剪阈值系数
* `--edge_temperature`: softmax 温度（`renorm_mode=softmax` 时生效）
* `--min_in_edges`: 每个节点至少保留的入边数（防止回溯断链）
* `--overlap_weight`: 轻量文本重叠证据与注意力融合权重
* `--failure_step_strategy`: failure step 选择策略（`final`/`max_anomaly`/`first_error`/`multi`）
* `--failure_sinks`: 多 sink 组合（默认 `final,max_anomaly,first_error`）
* `--sink_weight_final/--sink_weight_first_error/--sink_weight_anomaly`: 多 sink 权重
* `--train_prefix_only`: 训练时仅保留错误步之前的前缀（减少错误传播噪声）
* `--diagnose_k`: 输出失败诊断报告的样例数
* `--diagnose_dir`: 诊断报告输出目录
* `--who_mode`: who 预测模式（`global`/`neighborhood`）
* `--who_hops`: who 邻域投票 hop 数
* `--step_pred_mode`: step 预测模式（`model`/`scores`/`sim`/`agent_first`）
* `--agent_agg`: agent 聚合方式（`sum`/`logsumexp`/`max`/`topk`）
* `--agent_top_k`: top‑k 聚合时保留的步数
* `--agent_feat_dim`: 动态 agent 特征维度（one-hot bucket），`0` 表示自动
* `--agent_feat_dim_cap`: 动态维度上限（默认 32）
* `--split_strategy`: `random|deterministic`（随机或按 trace_id 固定切分）
* `--split_seed`: 随机切分种子
* `--model_type`: `mlp|gnn`

---

## 13.2 Aegis 数据集接入

`data/aegis` 下的 JSON 会被适配为统一结构：
- `conversation_history` → steps
- `faulty_agents` → who
- `when` 为 **faulty agent 首次出现的 step**（若找不到则 when 缺失）

示例命令（当前 best，带 seed）：
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
  --who_mode global
```

---

## 14. 配置（configs/default.yaml 建议字段）

* `adapter`: dataset 名称与策略
* `context_policy`: `all_history | sliding_window | role_filtered`
* `slm`: `model_name`, `dtype`, `max_length`, `attn_aggregate`
* `detector`: `alpha`, `beta`, `lambda_proto`, `topk_candidates`
* `graph`: `topM_in_edges`, `min_edge_weight`, `citation_boost`
* `tracer`: `lambda_score`, `influence_method`, `max_hops`
* `cache`: 路径、读写策略、版本号

---

## 15. 实现关键注意事项（会直接影响效果）

1. **token span 对齐必须严格**：Soft-IDG 的边权高度依赖 span 准确性
2. **图稀疏化**：每个节点保留 top-M 入边，防止噪声扩散
3. **缓存与复用**：注意力提取昂贵，必须缓存落盘
4. **适配层“保存 raw”**：避免数据集信息丢失导致后期无法补齐字段

---

## 16. 参考材料（已挂载）

* 课题介绍：强调级联故障、白盒诊断、高效归因与注意力替代复杂解析建图的方向 
* MASC：无监督 step-level 异常检测（reconstruction + prototype） 
* GraphTracer：信息依赖图（IDG）与根因回溯、路径解释 

---

## 17. 建议的最小可行 MVP（优先级顺序）

1. **Adapter 层 + Unified Schema + 校验器**（保证后续不返工）
2. **SLM Wrapper 输出 embedding + attention + spans（含缓存）**
3. **MASC-style anomaly score `s(t)`（简化版即可）** 
4. **Soft-IDG + 规则 tracer（λ*s + Influence）** 
5. **评测 + DOT/HTML 可视化**
6. 可选：GNN tracer、显式引用融合、更多消融

---

如果你希望 README 更“可执行交付”（例如补充：`report.json` 精确定义、统一 Schema 的 Pydantic 定义、WhoWhenAdapter 的字段映射模板、以及 `context_policy` 的默认选择建议），我可以在同一份 README 下继续补齐这些“工程接口级”细节。
