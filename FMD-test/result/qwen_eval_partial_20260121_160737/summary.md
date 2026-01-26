# Qwen3-max 真假判别结果（各100条随机采样）

- 总样本: 198 (错误: 2)
- 整体准确率: 0.8636

## 各类别指标
- 1_true: precision=0.8462, recall=0.8889, f1=0.8670, support=99
- 0_fake: precision=0.8830, recall=0.8384, f1=0.8601, support=99

## 混淆矩阵
- TP=88, TN=83, FP=16, FN=11

## 标签分布
- 真实标签: {'0': 100, '1': 100}
- 预测标签: {'0': 94, '1': 104, 'unknown': 2}

## Token 预估
- Prompt 前缀长度: 108 chars
- 总 Prompt 字符数: 915642
- 估算 tokens: 228911
