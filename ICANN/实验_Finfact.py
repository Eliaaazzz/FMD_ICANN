import json
import re
import time
from openai import OpenAI
from collections import Counter
from typing import Literal

# ============================================================
# 超参数设置
# ============================================================
NUM_SAMPLES = 3  # 【可调】要测试的样本数量，设为 -1 表示全部测试
MODEL_NAME = "qwen3-max"  # 可选: qwen-turbo, qwen-plus, qwen-max
API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"  # 替换为你的 API Key
DATA_PATH = "data/FinFact/finfact.json"  # 数据文件路径
SLEEP_INTERVAL = 0.1  # API 调用间隔（秒），避免限流

# ============================================================
# 初始化 Qwen 客户端（百炼平台兼容 OpenAI 接口）
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def load_data(filepath: str) -> list[dict]:
    """加载 JSON 格式的数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_ground_truth(item: dict) -> str:
    """从数据项中提取真实标签

    数据集的 label 字段值为: "true", "false", "NEI" (小写)
    统一转换为: "True", "False", "NEI"
    """
    label = item.get("label", "").strip().lower()
    if label == "true":
        return "True"
    elif label == "false":
        return "False"
    elif label == "nei":
        return "NEI"
    else:
        return "Unknown"


def format_evidence(evidence_list: list[dict]) -> str:
    """格式化证据列表为字符串"""
    if not evidence_list:
        return "无可用证据"

    formatted = []
    for i, ev in enumerate(evidence_list, 1):
        sentence = ev.get("sentence", "")
        hrefs = ev.get("hrefs", [])
        href_str = ", ".join(hrefs) if hrefs else "无来源链接"
        formatted.append(f"  证据{i}: {sentence}\n    来源: {href_str}")

    return "\n".join(formatted)


def build_instruction(item: dict) -> str:
    """根据数据集字段构建输入指令

    使用字段:
    - claim: 待验证的声明
    - sci_digest: 声明的核心摘要
    - justification: 事实核查的完整论证文本
    - evidence: 支撑论证的可溯源依据
    """
    claim = item.get("claim", "")

    # sci_digest 是列表，需要拼接
    sci_digest = item.get("sci_digest", [])
    if isinstance(sci_digest, list):
        sci_digest_text = " ".join(sci_digest)
    else:
        sci_digest_text = str(sci_digest)

    justification = item.get("justification", "")
    evidence = item.get("evidence", [])
    evidence_text = format_evidence(evidence)

    instruction = f"""请根据以下信息判断声明的真实性。

## 待验证声明
{claim}

## 声明摘要
{sci_digest_text}

## 事实核查论证
{justification}

## 支撑证据
{evidence_text}

请仔细分析上述信息，判断该声明的真实性。

输出要求：
1. 先给出分类预测标签，格式为 "Prediction: [标签]"，标签只能是以下三种之一：
   - True（声明属实）
   - False（声明虚假）
   - NEI（信息不足，无法判断）

2. 然后给出解释说明（Explanation），需包含：
   - 核心判定理由
   - 关键证据指向
   - 逻辑闭环分析

请按以下格式输出：
Prediction: [True/False/NEI]

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
                {"role": "system", "content": "你是一个专业的金融事实核查助手。请根据提供的声明、摘要、论证和证据信息，判断声明的真实性，并给出详细的解释说明。"},
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
    # 尝试匹配 "Prediction: True/False/NEI" 格式
    match = re.search(r"Prediction:\s*(True|False|NEI)", response_text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        if label == "true":
            return "True"
        elif label == "false":
            return "False"
        else:
            return "NEI"

    # 备选匹配：查找文本中的标签关键词
    response_lower = response_text.lower()
    if "预测" in response_lower or "判断" in response_lower or "结论" in response_lower:
        # 在关键词附近查找标签
        if re.search(r"(预测|判断|结论|标签)[：:]\s*(true|真实|属实)", response_lower):
            return "True"
        elif re.search(r"(预测|判断|结论|标签)[：:]\s*(false|虚假|不实)", response_lower):
            return "False"
        elif re.search(r"(预测|判断|结论|标签)[：:]\s*(nei|信息不足|无法判断)", response_lower):
            return "NEI"

    return "Unknown"


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """计算评估指标"""
    # 过滤掉 Unknown 的预测
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]

    correct = sum(1 for t, p in valid_pairs if t == p)
    total = len(valid_pairs)
    accuracy = correct / total if total > 0 else 0

    labels = ["True", "False", "NEI"]
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
    labels = ["True", "False", "NEI", "Unknown"]
    print("\n" + "=" * 60)
    print("混淆矩阵 (行=真实标签, 列=预测标签)")

    header = "        " + "".join(f"{label:>10}" for label in labels) + "     Total"
    print(header)
    print("-" * len(header))

    for true_label in ["True", "False", "NEI"]:  # 真实标签不包含 Unknown
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
    print("FinFact 数据集 Qwen 事实核查实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"测试样本数: {NUM_SAMPLES if NUM_SAMPLES > 0 else '全部'}")
    print()

    print("[1/4] 加载数据...")
    data = load_data(DATA_PATH)
    print(f"      总数据量: {len(data)} 条")

    if NUM_SAMPLES > 0:
        data = data[:NUM_SAMPLES]
    print(f"      测试数量: {len(data)} 条")

    # 打印标签分布
    label_dist = Counter([extract_ground_truth(item) for item in data])
    print(f"      标签分布: {dict(label_dist)}")

    y_true = []
    y_pred = []
    results = []

    print("\n[2/4] 开始调用 Qwen API 进行预测...")
    for i, item in enumerate(data):
        instruction = build_instruction(item)
        ground_truth = extract_ground_truth(item)

        print(f"      处理第 {i + 1}/{len(data)} 条...", end=" ")
        print(f"[{item.get('claim', '')[:30]}...]", end=" ")

        response = call_qwen_api(instruction)
        prediction = extract_prediction(response)

        y_true.append(ground_truth)
        y_pred.append(prediction)

        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth:>5}, 预测: {prediction:>7} {is_correct}")

        results.append({
            "index": i,
            "claim": item.get("claim", ""),
            "url": item.get("url", ""),
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
    for label in ["True", "False", "NEI"]:
        m = metrics[label]
        print(f"{label:>10} {m['precision']:>12.2%} {m['recall']:>12.2%} {m['f1']:>12.2%} {m['support']:>10}")

    print_confusion_matrix(y_true, y_pred)

    print("\n标签分布:")
    print(f"  真实标签: {dict(Counter(y_true))}")
    print(f"  预测标签: {dict(Counter(y_pred))}")

    # 保存结果
    output_file = "experiment_results_finfact.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": MODEL_NAME,
                "num_samples": len(data),
                "data_path": DATA_PATH
            },
            "metrics": metrics,
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()