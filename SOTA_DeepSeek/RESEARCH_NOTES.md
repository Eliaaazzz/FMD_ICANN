# FMD SOTA 项目研究笔记（2026-07-06）

## ★ 最终结果（2026-07-06，官方 FinFact 1304 测试集）

| 指标 | CATO-V (deepseek-v4-flash, training-free) | FMDLlama3 (SOTA) | Δ |
|---|---|---|---|
| Accuracy | **0.7722** | 0.7362 | +3.60 |
| Macro-Precision | **0.7458** | 0.6733 | +7.25 |
| Macro-Recall | **0.7493** | 0.6700 | +7.93 |
| Macro-F1 | **0.7454** | 0.6667 | **+7.87** |
| ROUGE-1 | **0.5229** | 0.4524 | +7.05 |
| ROUGE-2 | **0.4322** | 0.3498 | +8.24 |
| ROUGE-L | **0.4489** | 0.3773 | +7.16 |
| BERTScore | **0.7195** | 0.6756 | +4.39 |

**Table 3 全部 8 项指标超越，全程 training-free。**
评测协议与官方 `FMD-main/evaluation.ipynb` 逐项对齐：ROUGE = HF evaluate（无词干化）；
BERTScore = `model_type="bert-base-uncased"` 原始 F1（batch 16）。
（参考：若用 rouge_score 带词干化则 R1/R2/RL = 0.5296/0.4343/0.4507；
BERTScore roberta-large 原始 0.8922 / 基线重标定 0.3610——论文口径已确认为 bert-base。）
诚实备注：挑战赛综合分 avg(Micro-F1, R1) = (0.7722+0.5229)/2 = 0.648，
在 COLING25 榜单上约第 6/9（前排是微调到金标解释风格的系统）；
我们的对标对象是 FMDLlama 论文 Table 3（全部超越），且 Macro-F1 0.7454
仍是该测试集上已知最高。

分类 per-class F1：False 0.8235 / True 0.7930 / NEI 0.6196。
1304 条 0 解析失败 0 错误；3.7 分钟；成本 $0.39（¥2.8）。
运行目录：`results/test_v3_flash/`（v3 配置：kNN k=4 标签覆盖 + 语料约定 prompt +
自适应升级 + 评级引用验证映射 + NEI 门控；解释=抽取式 400 词）。

路径分布：single 1294 / escalated 7 / escalated+nei_gate 2 / single+rating_map 1
（v3 prompt 已让模型在源头正确映射，后处理仅兜底 1 次）。

制胜三板斧（消融自 dev199）：
1. v1→v2 语料标签约定 prompt（Mixture/Half True/Unproven→NEI 等）：
   dev ACC 0.6935→0.7889，macro-F1 0.6058→0.7502（NEI F1 0.33→0.61）
2. +评级引用验证映射（离线模拟 6 修复/0 破坏）：dev ACC→0.8191，macro-F1→0.7852
3. 检索 few-shot 全程提供标注口径校准

FinGuard（复现划分 2900/600/1500，官方划分未公开）：
- bge kNN k=9：0.8067 | **TF-IDF char(2-4) + LR C=4：0.9873 / macro-F1 0.9873**
- 论文参照：FMDLlama3 0.9947 / RoBERTa 0.9961（不同划分，仅方向可比）
- LLM zero-shot 在此任务本就弱（GPT-4o F1 0.51），未花 API 预算。

总花费：约 ¥3.8 / 预算 ¥10（余额 ¥6.2）。


## 1. 目标基准（FMDLlama, WWW Companion 2025, arXiv:2409.16452）

FinFact 三分类（测试集 1304 条，已恢复金标签：False 594 / True 455 / NEI 255）：

| 模型 | ACC | Macro-P | Macro-R | Macro-F1 | R1 | R2 | RL | BERTScore |
|---|---|---|---|---|---|---|---|---|
| **FMDLlama3 (SOTA)** | **0.7362** | 0.6733 | 0.6700 | **0.6667** | 0.4524 | 0.3498 | 0.3773 | 0.6756 |
| GPT-3.5-turbo | 0.7270 | 0.6635 | 0.6433 | 0.6380 | 0.2682 | — | — | — |
| GPT-4o-mini | 0.7132 | 0.7185 | 0.6368 | 0.6296 | — | — | — | — |
| GPT-4o | 0.6702 | 0.6596 | 0.6271 | 0.6283 | 0.2855 | — | — | 0.5668 |
| Llama3.1-8b-instruct | 0.6449 | — | — | 0.5383 | — | — | — | — |
| RoBERTa | 0.6822 | — | — | 0.5661 | — | — | — | — |

FinGuard 二分类（1500 条）：FMDLlama3 = 0.9947，RoBERTa = 0.9961（近饱和；FMDLlama 的
FinGuard 精确划分未公开，只有 FinFact 部分随 COLING25 挑战赛发布）。

## 2. 同测试集的真实天花板：COLING 2025 FMD 挑战赛（ACL 2025.finnlp-1.30）

- 同 1304 测试集（按 40% 公开/60% 私有排名），指标 = Micro-F1 与 ROUGE-1 的平均。
- 冠军 Dunamu ML：**Micro-F1 0.8467**，R1 0.8121（方法：MOCHEG 数据增强 + GPT-4 生成证据 +
  text-embedding-3-large kNN few-shot 选样 + **SFT Llama-3.1-8B**）。
- 前 6 名全部是微调模型；唯一 training-free 队伍 Capybara 垫底（Micro-F1 0.7221）。
- 该测试集上**无人公开报告过高于 0.6667 的 Macro-F1**（挑战赛只报 Micro-F1）。
- FMDLlama 作为 baseline 在挑战赛记分为 Micro-F1 0.7182 / R1 0.4502。

→ 我们的故事：**training-free（DeepSeek V4 + RAG 编排）超越微调 SOTA**。
  主目标：ACC > 0.7362、Macro-F1 > 0.6667；冲刺目标：ACC/Micro-F1 ≥ 0.8467。

## 3. 数据与协议（已对齐、已验证）

- 官方数据：HF `lzw1008/COLING25-FMD`；train 1953（= FMDLlama train1562+val391）、
  practice 600（⊂train，且有 12 条标签损坏——勿用作金标）、test 1304（隐藏标签）。
- 金标恢复：按 claim 精确匹配→归一化匹配→日期消歧，对 train 1953/1953 全对后应用于
  test → **1304/1304 全部恢复**（`data/finfact_test_gold.jsonl`）。
- 输入协议（与 FMDLlama/挑战赛一致）：claim + sci_digest（"Claim summaries"）+
  justification（"contextual information"）。**金标解释 = evidence 字段**（ROUGE/BERTScore 参照）。
- 长度：justification 平均 869 词 / p90 1503；gold evidence 平均 301 词 / 中位 224。
- 22.5% 的 justification 含显式裁决措辞（"we rate/our ruling"…）。
- 解释抽取上限（dev200）：justification 前 400 词 → R1 0.540 / R2 0.446 / RL 0.461，
  **裸抽取已超 FMDLlama 全部解释指标** → 解释模块用"判决锚定抽取式改写"。

## 4. DeepSeek V4（2026-04-24 Preview，7 月中正式版）

- API：`base_url=https://api.deepseek.com`，模型 `deepseek-v4-pro`（1.6T MoE/49B 激活）、
  `deepseek-v4-flash`（284B/13B）。1M 上下文 / 384K 输出。
- 思考模式：`extra_body={"thinking":{"type":"enabled"}}` 或 `reasoning_effort`。
- 价格（每百万 token，非峰时）：flash 输入 $0.14（缓存命中 $0.0028）/输出 $0.28；
  pro 输入 $0.435/输出 $0.87。峰时×2（北京 9-12、14-18 点）。
- 旧名 deepseek-chat/reasoner → v4-flash 非思考/思考，2026-07-24 退役。

## 5. 方法设计（工作名 CATO-V：CATO 框架的 verdict 实例化）

每条测试 claim：
1. **混合检索 kNN**：BM25 + bge-small-en-v1.5 余弦，RRF 融合，取 top-4 训练集带标签
   同源示例（教会模型该语料的标注口径——Dunamu 冠军的关键招，我们免微调复刻）。
2. **结构化判决**（v4 非思考, JSON）：操作性标签定义 + 示例 + 全文上下文 →
   {key_assertion, evidence_for/against, sufficiency, prediction, confidence}。
3. **自一致性 3 票**（temp 0 / 0.8 / 0.8；前缀缓存使追加票近乎免费）。
4. **分歧仲裁**（v4 思考模式）：不一致时把三票意见交仲裁员，反事实检验后终裁
   （MIND/LoCal 的"分歧才付费"模式）。
5. **NEI 充分性门控**（可调）：多数票 sufficiency=insufficient 时强制 NEI
   （FinVet 证实这是 NEI 召回的最大杠杆）。
6. **解释生成**：以最终标签为锚，从 justification 抽取+改写，kNN 同标签邻居的
   gold evidence 作风格参照，目标 300-400 词。

消融计划（论文用）：−kNN(k=0)、k∈{2,4,8}、−自一致性、−仲裁、−NEI门控、
flash vs pro、思考 vs 非思考。

## 6. 支撑文献要点（均已精读，注意磁盘文件名与内容错位）

- 检索增强 few-shot 选样 = 冠军关键（Dunamu, 2025.finnlp-1.30）。
- 证据充分性门控→NEI + 置信度加权投票 +10.4%（FinVet, arXiv:2510.11654）。
- 分解-检索-验证-逻辑聚合 + 反事实评估器（LoCal, WWW'25 投稿版）。
- 判决通道仲裁/顾问模式（Claim Veracity Assessment, COLING 2025）。
- 分歧才启用 judge、检索样本先蒸馏成规则再用（MIND, arXiv:2507.06908）。
- 辩论对"全都像真的"偏置的纠正 + DeepSeek 可作骨干（TED/TruEDebate, SIGIR 2025）。
- 短辩论/层级裁决防错误放大（Who's the Mole, arXiv:2507.04724）。
- 数值验证工具对金融声明重要（Multi-Tool Agent, GAIB 2025）。
- ClaimCheck（arXiv:2510.01226）：Qwen3-4B + 好管线 > GPT-4o——管线设计>模型体积。

## 7. 文件与位置

- `SOTA_DeepSeek/recover_test_labels.py` — 金标恢复（已验证 1953/1953）
- `SOTA_DeepSeek/build_index.py` — 嵌入索引 + dev200 划分
- `SOTA_DeepSeek/run_pipeline.py` — 主管线（断点续跑：同 tag 重跑自动跳过已完成）
- `SOTA_DeepSeek/evaluate.py` — 全指标评测 + 与 FMDLlama Table 3 对照
- `SOTA_DeepSeek/data/finfact_test_gold.jsonl` — 1304 金标测试集
- `SOTA_DeepSeek/data/finfact_dev_gold.jsonl` — dev200（含 gold_evidence）
- 运行成本估算：dev200 调参一轮 ≈ $0.3（flash）；test1304 一遍 ≈ $2.5（flash）/ $8（pro）
