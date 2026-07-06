# FMD_ICANN — Financial Misinformation Detection (ICANN)

用 **DeepSeek V4（training-free，无任何微调）** 在 FMDLlama（WWW 2025,
arXiv:2409.16452）的官方 FinFact 基准上超越其全部 SOTA 指标。

## 最终结果（官方 FinFact 1304 测试集，官方评测协议）

| 指标 | CATO-V（本仓库，deepseek-v4-flash） | FMDLlama3（微调 SOTA） | Δ |
|---|---|---|---|
| Accuracy | **0.7722** | 0.7362 | +3.60 |
| Macro-Precision | **0.7458** | 0.6733 | +7.25 |
| Macro-Recall | **0.7493** | 0.6700 | +7.93 |
| Macro-F1 | **0.7454** | 0.6667 | **+7.87** |
| ROUGE-1 | **0.5229** | 0.4524 | +7.05 |
| ROUGE-2 | **0.4322** | 0.3498 | +8.24 |
| ROUGE-L | **0.4489** | 0.3773 | +7.16 |
| BERTScore | **0.7195** | 0.6756 | +4.39 |

- 测试集 = COLING 2025 FMD 挑战赛官方隐藏测试集（1304 条，金标签已通过
  claim 匹配恢复，匹配方法在 1953 条训练集上 100% 验证）。
- 评测协议与官方 `FMD-main/evaluation.ipynb` 对齐（ROUGE 无词干化，
  BERTScore = bert-base-uncased 原始 F1）。
- 同一测试集的挑战赛前六名全部为微调模型；本方法免训练，
  全量 1304 条推理耗时 3.7 分钟，API 成本约 ¥2.8。
- 附带 FinGuard（复现划分）：TF-IDF char n-gram + LR = 0.9873（零 API 成本）。

## 方法（CATO-V，详见 `SOTA_DeepSeek/RESEARCH_NOTES.md`）

1. 混合检索 kNN few-shot（BM25 + bge-small，RRF 融合，标签覆盖修补）——
   免微调学习语料标注口径
2. 结构化判决 JSON：语料标签约定显式化（Snopes/PolitiFact "Mixture /
   Half True / Unproven" → NEI，"Mostly True" → True；训练集统计验证
   tail-Mixture 100%→NEI）+ 编辑评级引用字段 + resolution 字段
3. 自适应升级：低置信 / 自相矛盾 / NEI 线索词 → 追加 2 票自一致性
   （前缀缓存命中 93%，追加票近乎免费）
4. 分歧 → DeepSeek V4 思考模式仲裁（多轮续写复用缓存前缀）
5. 评级引用验证映射（引用必须在原文中出现才生效，防幻觉）+ NEI 门控
6. 解释生成：判决锚定抽取式（零额外成本）

## 目录

- `SOTA_DeepSeek/` — **本次 SOTA 管线**：代码、恢复的金标测试集、全部运行结果、
  研究笔记与复现 README
- `ICANN/` — 早期 prompt 基线实验（qwen3-max）与 CATO/DAG 数据库构建脚本
- `FMD-main/` — FMDLlama 官方仓库镜像（含官方 evaluation.ipynb）
- `Fin-Fact-FinFact/` — Fin-Fact 原始数据集（finfact.json，3369 条）
- `FMD_ICANN-paper-/` — 早期 FMDLlama 4bit 本地复现尝试
- `Financial-Truth-Guard-main/`、`FMD-test/`、`Edition_1/` — FinGuard 数据与历史实验
- `note/` — 论文阅读笔记
- `FMD.md`、`DAG-agent-虚假信息检测-检索-方案.md` — 实验记录与方案草稿

## 复现

```bash
cd SOTA_DeepSeek
pip install openai rank_bm25 rouge_score bert_score sentence-transformers scikit-learn numpy tqdm
# 原始挑战赛数据（41MB，未入库）：
#   https://huggingface.co/datasets/lzw1008/COLING25-FMD → data/coling25_fmd/
python recover_test_labels.py   # 恢复官方测试集金标签
python build_index.py           # 构建检索索引（CPU 即可）
export DEEPSEEK_API_KEY=sk-...
python run_pipeline.py --split test --tag my_run --budget-usd 1.0
python evaluate.py --tag my_run --bertscore
```

论文阅读 PDF 与原始挑战赛数据因体积/版权未入库；见各目录 README 与上方链接。
