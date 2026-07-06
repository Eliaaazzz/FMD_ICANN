# CATO-V: Training-free FinFact SOTA with DeepSeek V4

在官方 FinFact 1304 测试集（FMDLlama / COLING25-FMD 挑战赛同款）上，
免训练管线 ACC **0.7722** / Macro-F1 **0.7454**，超越 FMDLlama3
（0.7362 / 0.6667）。全量运行成本约 ¥2.8。详见 `RESEARCH_NOTES.md`。

## 复现步骤

```bash
pip install openai rank_bm25 rouge_score bert_score sentence-transformers scikit-learn numpy tqdm

# 1. 恢复官方测试集金标签（需 Fin-Fact-FinFact/finfact.json + data/coling25_fmd/）
python recover_test_labels.py

# 2. 构建检索索引与 dev 划分（bge-small-en-v1.5, CPU 即可）
python build_index.py

# 3. 跑管线（需环境变量 DEEPSEEK_API_KEY）
python run_pipeline.py --split dev  --tag dev_run              # dev199 调参
python run_pipeline.py --split test --tag test_run --budget-usd 1.0   # 全量测试

# 4. 评测（ROUGE + BERTScore + 与 FMDLlama Table 3 对照）
python evaluate.py --tag test_run --bertscore

# 5. FinGuard（零 API 成本基线）
python finguard_knn.py
```

断点续跑：同一 `--tag` 重跑会自动跳过已完成条目；`--budget-usd` 超限自动熔断。

## 管线结构（run_pipeline.py）

1. 混合检索 kNN few-shot（BM25 + bge, RRF 融合，标签覆盖修补）
2. 结构化判决 JSON（语料标签约定 + 评级引用字段 + resolution 字段）
3. 自适应升级：低置信/自相矛盾/NEI 线索词 → 追加 2 票（前缀缓存近免费）
4. 分歧 → V4 思考模式仲裁（多轮续写复用缓存）
5. 评级引用验证映射（quote 必须在原文出现才生效）+ NEI 门控
6. 解释：判决锚定抽取式（justification 前 400 词）
