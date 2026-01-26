import json
import re
import time
import random
import pandas as pd
from openai import OpenAI
from collections import Counter

# ============================================================
# 超参数设置
# ============================================================
NUM_SAMPLES = 10  # 【可调】要测试的样本总数（从TRUE和FAKE各抽取一半）
RANDOM_SEED = 42  # 【可调】随机种子，保证可复现
MODEL_NAME = "qwen3-max"  # 可选: qwen-turbo, qwen-plus, qwen-max
API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"  # 替换为你的 API Key
TRUE_DATA_PATH = "data/FinGuard/Finance_TRUE.csv"  # TRUE数据文件路径
FAKE_DATA_PATH = "data/FinGuard/Finance_FAKE.csv"  # FAKE数据文件路径
SLEEP_INTERVAL = 0.1  # API 调用间隔（秒），避免限流

# ============================================================
# 初始化 Qwen 客户端（百炼平台兼容 OpenAI 接口）
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def load_and_sample_data(true_path: str, fake_path: str, num_samples: int, seed: int) -> list[dict]:
    """加载CSV数据并随机抽样

    Args:
        true_path: TRUE数据文件路径
        fake_path: FAKE数据文件路径
        num_samples: 总抽样数量（从TRUE和FAKE各抽取一半）
        seed: 随机种子

    Returns:
        混合后的数据列表，每个元素包含 'text' 和 'label' 字段
    """
    random.seed(seed)

    # 加载CSV文件
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    print(f"      TRUE数据总量: {len(df_true)} 条")
    print(f"      FAKE数据总量: {len(df_fake)} 条")

    # 计算每类抽取数量
    samples_per_class = num_samples // 2
    samples_true = min(samples_per_class, len(df_true))
    samples_fake = min(samples_per_class, len(df_fake))

    # 如果总样本数为奇数，多抽一个TRUE
    if num_samples % 2 == 1:
        samples_true = min(samples_true + 1, len(df_true))

    # 随机抽样
    true_indices = random.sample(range(len(df_true)), samples_true)
    fake_indices = random.sample(range(len(df_fake)), samples_fake)

    data = []

    # 添加TRUE样本
    for idx in true_indices:
        row = df_true.iloc[idx]
        data.append({
            "text": row["text"],
            "label": "True",
            "source_file": "TRUE",
            "original_index": idx
        })

    # 添加FAKE样本
    for idx in fake_indices:
        row = df_fake.iloc[idx]
        data.append({
            "text": row["text"],
            "label": "False",
            "source_file": "FAKE",
            "original_index": idx
        })

    # 打乱顺序
    random.shuffle(data)

    return data


def extract_ground_truth(item: dict) -> str:
    """从数据项中提取真实标签

    数据集的 label 字段值为: "True", "False"
    """
    return item.get("label", "Unknown")


def build_instruction(item: dict) -> str:
    """根据数据集字段构建输入指令

    使用字段:
    - text: 待验证的新闻文本
    """
    text = item.get("text", "")

    # 截取文本（如果太长）
    max_length = 3000
    if len(text) > max_length:
        text = text[:max_length] + "...[文本已截断]"

    instruction = f"""请根据以下新闻文本判断其真实性。

## 待验证新闻
{text}

请仔细分析上述新闻内容，判断该新闻的真实性。你需要考虑：
1. 新闻内容是否符合常识和逻辑
2. 语言表达是否专业规范
3. 是否存在明显的夸大、煽情或误导性内容
4. 信息来源是否可靠

输出要求：
1. 先给出分类预测标签，格式为 "Prediction: [标签]"，标签只能是以下两种之一：
   - True（新闻属实/可信）
   - False（新闻虚假/不可信）

2. 然后给出解释说明（Explanation），需包含：
   - 核心判定理由
   - 关键特征分析
   - 可信度评估

请按以下格式输出：
Prediction: [True/False]

Explanation:
[你的解释说明]
"""
    return instruction


def call_qwen_api(instruction: str) -> str:
    """调用 Qwen API 获取预测结果"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的新闻事实核查助手。请根据提供的新闻文本，判断其真实性，并给出详细的解释说明。"},
                {"role": "user", "content": instruction},
            ],
            temperature=0.1,  # 低温度使输出更稳定
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用出错: {e}")
        return ""


def extract_prediction(response_text: str) -> str:
    """从模型回复中提取预测标签"""
    # 尝试匹配 "Prediction: True/False" 格式
    match = re.search(r"Prediction:\s*(True|False)", response_text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        if label == "true":
            return "True"
        elif label == "false":
            return "False"

    # 备选匹配：查找文本中的标签关键词
    response_lower = response_text.lower()
    if "预测" in response_lower or "判断" in response_lower or "结论" in response_lower:
        # 在关键词附近查找标签
        if re.search(r"(预测|判断|结论|标签)[：:]\s*(true|真实|属实|可信)", response_lower):
            return "True"
        elif re.search(r"(预测|判断|结论|标签)[：:]\s*(false|虚假|不实|不可信)", response_lower):
            return "False"

    return "Unknown"


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """计算评估指标"""
    # 过滤掉 Unknown 的预测
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]

    correct = sum(1 for t, p in valid_pairs if t == p)
    total = len(valid_pairs)
    accuracy = correct / total if total > 0 else 0

    labels = ["True", "False"]
    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "unknown_count": len(y_true) - len(valid_pairs)
    }

    # 计算各类别指标
    for label in labels:
        tp = sum(1 for t, p in valid_pairs if t == label and p == label)
        fp = sum(1 for t, p in valid_pairs if t != label and p == label)
        fn = sum(1 for t, p in valid_pairs if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label)
        }

    # 计算宏平均 F1
    valid_f1s = [metrics[label]["f1"] for label in labels if metrics[label]["support"] > 0]
    metrics["macro_f1"] = sum(valid_f1s) / len(valid_f1s) if valid_f1s else 0

    # ============================================================
    # 计算整体的 Precision, Recall, F1 (宏平均)
    # ============================================================
    valid_precisions = [metrics[label]["precision"] for label in labels if metrics[label]["support"] > 0]
    valid_recalls = [metrics[label]["recall"] for label in labels if metrics[label]["support"] > 0]

    metrics["macro_precision"] = sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0
    metrics["macro_recall"] = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0

    return metrics


def print_confusion_matrix(y_true: list[str], y_pred: list[str]):
    """打印混淆矩阵"""
    labels = ["True", "False", "Unknown"]
    print("\n" + "=" * 60)
    print("混淆矩阵 (行=真实标签, 列=预测标签)")

    header = "        " + "".join(f"{label:>10}" for label in labels) + "     Total"
    print(header)
    print("-" * len(header))

    for true_label in ["True", "False"]:  # 真实标签不包含 Unknown
        row = f"{true_label:>8}"
        row_total = 0
        for pred_label in labels:
            count = sum(1 for t, p in zip(y_true, y_pred) if t == true_label and p == pred_label)
            row += f"{count:>10}"
            row_total += count
        row += f"{row_total:>10}"
        print(row)

    # 打印预测总计行
    print("-" * len(header))
    total_row = "   Total"
    grand_total = 0
    for pred_label in labels:
        count = sum(1 for p in y_pred if p == pred_label)
        total_row += f"{count:>10}"
        grand_total += count
    total_row += f"{grand_total:>10}"
    print(total_row)


def main():
    print("=" * 60)
    print("Finance TRUE/FAKE 数据集 Qwen 事实核查实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"测试样本数: {NUM_SAMPLES}")
    print()

    print("[1/4] 加载数据并随机抽样...")
    data = load_and_sample_data(TRUE_DATA_PATH, FAKE_DATA_PATH, NUM_SAMPLES, RANDOM_SEED)
    print(f"      实际测试数量: {len(data)} 条")

    # 打印标签分布
    label_dist = Counter([item["label"] for item in data])
    print(f"      标签分布: {dict(label_dist)}")

    y_true = []
    y_pred = []
    results = []

    print("\n[2/4] 开始调用 Qwen API 进行预测...")
    for i, item in enumerate(data):
        instruction = build_instruction(item)
        ground_truth = extract_ground_truth(item)

        # 显示文本预览（前50个字符）
        text_preview = item.get("text", "")[:50].replace("\n", " ")
        print(f"      处理第 {i + 1}/{len(data)} 条...", end=" ")
        print(f"[{text_preview}...]", end=" ")

        response = call_qwen_api(instruction)
        prediction = extract_prediction(response)

        y_true.append(ground_truth)
        y_pred.append(prediction)

        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth:>5}, 预测: {prediction:>7} {is_correct}")

        results.append({
            "index": i,
            "text_preview": item.get("text", "")[:200],
            "source_file": item.get("source_file", ""),
            "original_index": item.get("original_index", -1),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": ground_truth == prediction,
            "response": response[:500] + "..." if len(response) > 500 else response,
        })

        time.sleep(SLEEP_INTERVAL)

    print("\n[3/4] 计算评估指标...")
    metrics = calculate_metrics(y_true, y_pred)

    print("\n[4/4] 实验结果")
    print("=" * 60)

    # 打印整体指标
    print("【整体指标】")
    print(f"  Accuracy (ACC):   {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Precision (PRE):  {metrics['macro_precision']:.2%}")
    print(f"  Recall (REC):     {metrics['macro_recall']:.2%}")
    print(f"  F1-Score (F1):    {metrics['macro_f1']:.2%}")

    if metrics['unknown_count'] > 0:
        print(f"  无法识别预测: {metrics['unknown_count']} 条")

    print("\n【各类别指标】")
    print(f"{'类别':>10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Support':>10}")
    print("-" * 56)
    for label in ["True", "False"]:
        m = metrics[label]
        print(f"{label:>10} {m['precision']:>12.2%} {m['recall']:>12.2%} {m['f1']:>12.2%} {m['support']:>10}")

    print_confusion_matrix(y_true, y_pred)

    print("\n标签分布:")
    print(f"  真实标签: {dict(Counter(y_true))}")
    print(f"  预测标签: {dict(Counter(y_pred))}")

    # 保存结果
    output_file = "experiment_results_finance.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": MODEL_NAME,
                "num_samples": len(data),
                "random_seed": RANDOM_SEED,
                "true_data_path": TRUE_DATA_PATH,
                "fake_data_path": FAKE_DATA_PATH
            },
            "metrics": metrics,
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()