import json
import re
import time
from datetime import datetime
from openai import OpenAI
from collections import Counter
from typing import Literal
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()

# ============================================================
# 超参数设置
# ============================================================
NUM_SAMPLES = 100  # 【可调】要测试的样本数量，设为 -1 表示全部测试
MODEL_NAME = "qwen3-max"  # 可选: qwen-turbo, qwen-plus, qwen-max
API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"  # 替换为你的 API Key
DATA_PATH = SCRIPT_DIR / "data/FinFact/finfact.json"  # 数据文件路径
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


def build_instruction(item: dict) -> tuple[str, dict]:
    """根据数据集字段构建输入指令，符合指定的任务格式
    
    Returns:
        instruction: 完整的指令文本
        input_parts: 输入各部分的字典，用于结果记录
    """

    claim = item.get("claim", "")

    # sci_digest 作为 claim summaries，可能是列表
    sci_digest = item.get("sci_digest", [])
    sci_digest_text = " ".join(sci_digest) if isinstance(sci_digest, list) else str(sci_digest)

    justification = item.get("justification", "")
    evidence = item.get("evidence", [])
    evidence_text = format_evidence(evidence)

    contextual = f"{justification}\n\n证据:\n{evidence_text}".strip()

    task = "Please determine whether the claim is 0. False, 1. True, or 2. Not Enough Information (NEI) based on contextual information, and provide an appropriate explanation."
    prediction_format = "[0. False, 1. True, or 2. NEI]"
    explanation_format = "[Explain why the above prediction was made]"

    instruction = f"""Task: {task}
The answer needs to use the following format:
Prediction: {prediction_format}
Explanation: {explanation_format}
Claim: {claim}
Claim summaries: {sci_digest_text}
Contextual information: {contextual}
"""

    input_parts = {
        "Task": task,
        "Prediction_format": prediction_format,
        "Explanation_format": explanation_format,
        "Claim": claim,
        "Claim_summaries": sci_digest_text,
        "Contextual_information": contextual
    }

    return instruction, input_parts


def call_qwen_api(instruction: str) -> str:
    """调用 Qwen API 获取预测结果"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
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
    """从模型回复中提取预测标签，支持数字 0/1/2"""
    match = re.search(r"Prediction:\s*(0|1|2)", response_text, re.IGNORECASE)
    if match:
        val = match.group(1)
        return {"0": "False", "1": "True", "2": "NEI"}.get(val, "Unknown")

    # 备选匹配：英文标签
    match_text = re.search(r"Prediction:\s*(True|False|NEI)", response_text, re.IGNORECASE)
    if match_text:
        label = match_text.group(1).lower()
        if label == "true":
            return "True"
        if label == "false":
            return "False"
        if label == "nei":
            return "NEI"

    response_lower = response_text.lower()
    if "预测" in response_lower or "判断" in response_lower or "结论" in response_lower:
        if re.search(r"(预测|判断|结论|标签)[：:]\s*(0|false|虚假|不实)", response_lower):
            return "False"
        if re.search(r"(预测|判断|结论|标签)[：:]\s*(1|true|真实|属实)", response_lower):
            return "True"
        if re.search(r"(预测|判断|结论|标签)[：:]\s*(2|nei|信息不足|无法判断)", response_lower):
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
        instruction, input_parts = build_instruction(item)
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
            "url": item.get("url", ""),
            "ground_truth": ground_truth,
            "correct": ground_truth == prediction,
            # 输入部分
            "input": {
                "Task": input_parts["Task"],
                "Prediction_format": input_parts["Prediction_format"],
                "Explanation_format": input_parts["Explanation_format"],
                "Claim": input_parts["Claim"],
                "Claim_summaries": input_parts["Claim_summaries"],
                "Contextual_information": input_parts["Contextual_information"]
            },
            # 输出部分
            "output": {
                "Prediction": prediction,
                "Explanation": response
            }
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
    result_dir = SCRIPT_DIR / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = result_dir / f"experiment_results_finfact_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": MODEL_NAME,
                "num_samples": len(data),
                "data_path": str(DATA_PATH)
            },
            "metrics": metrics,
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()