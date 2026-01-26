"""
Fin-Fact 数据集 + Qwen API 事实核查实验
========================================
使用阿里云百炼平台的 Qwen 大模型对金融声明进行真实性判断
"""

import json
import re
import time
from openai import OpenAI
from collections import Counter
from typing import Literal

# ============================================================
# 超参数设置
# ============================================================
NUM_SAMPLES = -1  # 【可调】要测试的样本数量，设为 -1 表示全部测试
MODEL_NAME = "qwen3-max"  # 可选: qwen-turbo, qwen-plus, qwen-max
API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"  # 替换为你的 API Key
DATA_PATH = "data/FMD/FMD_test.json"  # 数据文件路径
SLEEP_INTERVAL = 0.1  # API 调用间隔（秒），避免限流

# ============================================================
# 初始化 Qwen 客户端（百炼平台兼容 OpenAI 接口）
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def load_data(filepath: str) -> list[dict]:
    """加载 JSON Lines 格式的数据"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_ground_truth(output_text: str) -> str:
    """从 output 字段提取真实标签"""
    # 匹配 "Prediction: True" 或 "Prediction: False" 或 "Prediction: NEI"
    match = re.search(r"Prediction:\s*(True|False|NEI)", output_text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        # 标准化
        if label == "true":
            return "True"
        elif label == "false":
            return "False"
        else:
            return "NEI"
    return "Unknown"


def call_qwen_api(instruction: str) -> str:
    """调用 Qwen API 获取预测结果"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的事实核查助手。请根据提供的信息判断声明的真实性。",
                },
                {"role": "user", "content": instruction},
            ],
            temperature=0.1,  # 低温度使输出更稳定
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用出错: {e}")
        return ""


def extract_prediction(response_text: str) -> str:
    """从模型回复中提取预测标签"""
    # 尝试匹配标准格式
    match = re.search(r"Prediction:\s*(True|False|NEI)", response_text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        if label == "true":
            return "True"
        elif label == "false":
            return "False"
        else:
            return "NEI"

    # 如果没有标准格式，尝试在文本中查找关键词
    response_lower = response_text.lower()
    if "true" in response_lower and "false" not in response_lower:
        return "True"
    elif "false" in response_lower and "true" not in response_lower:
        return "False"
    elif "nei" in response_lower or "not enough" in response_lower:
        return "NEI"

    return "Unknown"


def calculate_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict:
    """计算评估指标"""
    # 过滤掉 Unknown 的预测
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]

    # 整体准确率
    correct = sum(1 for t, p in valid_pairs if t == p)
    total = len(valid_pairs)
    accuracy = correct / total if total > 0 else 0

    # 各类别统计
    labels = ["True", "False", "NEI"]
    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "unknown_count": len(y_true) - len(valid_pairs)
    }

    for label in labels:
        tp = sum(1 for t, p in valid_pairs if t == label and p == label)
        fp = sum(1 for t, p in valid_pairs if t != label and p == label)
        fn = sum(1 for t, p in valid_pairs if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label),
        }

    # ============================================================
    # 计算整体的 Precision, Recall, F1 (宏平均)
    # ============================================================
    valid_precisions = [metrics[label]["precision"] for label in labels if metrics[label]["support"] > 0]
    valid_recalls = [metrics[label]["recall"] for label in labels if metrics[label]["support"] > 0]
    valid_f1s = [metrics[label]["f1"] for label in labels if metrics[label]["support"] > 0]

    metrics["macro_precision"] = sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0
    metrics["macro_recall"] = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0
    metrics["macro_f1"] = sum(valid_f1s) / len(valid_f1s) if valid_f1s else 0

    return metrics


def print_confusion_matrix(y_true: list[str], y_pred: list[str]):
    """打印混淆矩阵"""
    labels = ["True", "False", "NEI", "Unknown"]
    print("\n" + "=" * 60)
    print("混淆矩阵 (行=真实标签, 列=预测标签)")
    print("=" * 60)

    # 表头
    header = "        " + "".join(f"{label:>10}" for label in labels) + "     Total"
    print(header)
    print("-" * len(header))

    for true_label in ["True", "False", "NEI"]:
        row = f"{true_label:>8}"
        row_total = 0
        for pred_label in labels:
            count = sum(
                1 for t, p in zip(y_true, y_pred) if t == true_label and p == pred_label
            )
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
    print("FMD 数据集 Qwen 事实核查实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"测试样本数: {NUM_SAMPLES if NUM_SAMPLES > 0 else '全部'}")
    print()

    # 1. 加载数据
    print("[1/4] 加载数据...")
    data = load_data(DATA_PATH)
    print(f"      总数据量: {len(data)} 条")

    # 2. 抽样
    if NUM_SAMPLES > 0:
        data = data[:NUM_SAMPLES]
    print(f"      测试数量: {len(data)} 条")

    # 打印标签分布
    label_dist = Counter([extract_ground_truth(item["output"]) for item in data])
    print(f"      标签分布: {dict(label_dist)}")

    # 3. 遍历预测
    print("\n[2/4] 开始调用 Qwen API 进行预测...")
    y_true = []
    y_pred = []
    results = []

    for i, item in enumerate(data):
        instruction = item["instruction"]
        ground_truth = extract_ground_truth(item["output"])

        print(f"      处理第 {i+1}/{len(data)} 条...", end=" ")

        # 调用 API
        response = call_qwen_api(instruction)
        prediction = extract_prediction(response)

        y_true.append(ground_truth)
        y_pred.append(prediction)

        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth:>5}, 预测: {prediction:>7} {is_correct}")

        results.append(
            {
                "index": i,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": ground_truth == prediction,
                "response": response[:200] + "..." if len(response) > 200 else response,
            }
        )

        # 休眠避免限流
        time.sleep(SLEEP_INTERVAL)

    # 4. 计算指标
    print("\n[3/4] 计算评估指标...")
    metrics = calculate_metrics(y_true, y_pred)

    # 5. 输出结果
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
    for label in ["True", "False", "NEI"]:
        m = metrics[label]
        print(
            f"{label:>10} {m['precision']:>12.2%} {m['recall']:>12.2%} {m['f1']:>12.2%} {m['support']:>10}"
        )

    # 混淆矩阵
    print_confusion_matrix(y_true, y_pred)

    # 标签分布
    print("\n标签分布:")
    print(f"  真实标签: {dict(Counter(y_true))}")
    print(f"  预测标签: {dict(Counter(y_pred))}")

    # 保存详细结果
    output_file = "experiment_results_fmd.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model": MODEL_NAME,
                    "num_samples": len(data),
                    "data_path": DATA_PATH
                },
                "metrics": metrics,
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()