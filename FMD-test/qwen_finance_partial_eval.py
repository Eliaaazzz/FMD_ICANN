import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from math import ceil
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

PROMPT_PREFIX = (
    "Task: Please determine whether the text is 0. Fake or 1. True. "
    "Answer directly without explanations.  Text: "
)


def print_stage(stage: str) -> None:
    """打印当前运行阶段"""
    print(f"\n{'='*50}")
    print(f"  {stage}")
    print(f"{'='*50}")


def reservoir_sample_csv(path: str, text_col: str, k: int, seed: int) -> List[str]:
    """蓄水池采样，随机抽取 k 条记录"""
    rng = random.Random(seed)
    sample: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found in {path}. Columns: {reader.fieldnames}")
        for i, row in enumerate(reader, start=1):
            text = row.get(text_col, "") or ""
            if i <= k:
                sample.append(text)
            else:
                j = rng.randint(1, i)
                if j <= k:
                    sample[j - 1] = text
    return sample


def estimate_tokens_from_chars(chars: int) -> int:
    return int(ceil(chars / 4))


def parse_prediction(text: str) -> Optional[int]:
    if not text:
        return None
    text = text.strip()
    for ch in text:
        if ch == "0":
            return 0
        if ch == "1":
            return 1
    return None


def call_model(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> Tuple[Optional[int], Optional[str]]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            content = resp.choices[0].message.content
            pred = parse_prediction(content)
            return pred, content
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                return None, f"ERROR: {exc}"
            time.sleep(1.5 * (2 ** attempt))
    return None, None


def update_metrics(metrics: Dict[str, int], y_true: int, y_pred: Optional[int]) -> None:
    if y_pred is None:
        metrics["errors"] += 1
        return
    if y_true == 1 and y_pred == 1:
        metrics["tp"] += 1
    elif y_true == 0 and y_pred == 0:
        metrics["tn"] += 1
    elif y_true == 0 and y_pred == 1:
        metrics["fp"] += 1
    elif y_true == 1 and y_pred == 0:
        metrics["fn"] += 1


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def compute_report(metrics: Dict[str, int]) -> Dict[str, object]:
    tp = metrics["tp"]
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    errors = metrics["errors"]

    total = tp + tn + fp + fn
    
    # 整体指标
    accuracy = safe_div(tp + tn, total)
    
    # 宏平均 (Macro Average)
    precision_pos = safe_div(tp, tp + fp)
    recall_pos = safe_div(tp, tp + fn)
    f1_pos = safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = safe_div(tn, tn + fn)
    recall_neg = safe_div(tn, tn + fp)
    f1_neg = safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    
    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        "overall_metrics": {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        },
        "per_class": {
            "1_true": {
                "precision": precision_pos,
                "recall": recall_pos,
                "f1": f1_pos,
                "support": tp + fn,
            },
            "0_fake": {
                "precision": precision_neg,
                "recall": recall_neg,
                "f1": f1_neg,
                "support": tn + fp,
            },
        },
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "errors": errors,
        "total_evaluated": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-max on finance真假数据集 (随机各100条)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  uv run --python .venv/Scripts/python.exe qwen_finance_partial_eval.py --api_key YOUR_API_KEY
  uv run --python .venv/Scripts/python.exe qwen_finance_partial_eval.py --api_key YOUR_API_KEY --sample_size 50
        """
    )
    # 获取脚本所在目录，用于构建相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_true_csv = os.path.join(script_dir, "Data_FinGuard", "Finance_TRUE.csv")
    default_fake_csv = os.path.join(script_dir, "Data_FinGuard", "Finance_FAKE.csv")
    
    parser.add_argument("--true_csv", default=default_true_csv)
    parser.add_argument("--fake_csv", default=default_fake_csv)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model", default="qwen3-max", help="使用的模型名称 (默认: qwen3-max)")
    parser.add_argument("--base_url", default=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api_key", required=False, default=os.getenv("QWEN_API_KEY", ""), help="Qwen API Key (必需)")
    parser.add_argument("--requests_per_minute", type=float, default=0.0)
    parser.add_argument("--sample_size", type=int, default=100, help="每类采样数量 (默认: 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: API key is empty. Please set --api_key or env QWEN_API_KEY.")
        return 2
    if not args.base_url:
        print("ERROR: Base URL is empty. Please set --base_url or env QWEN_BASE_URL.")
        return 2

    # 打印配置信息
    print_stage("配置信息")
    print(f"  模型名称: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  每类采样数: {args.sample_size}")
    print(f"  随机种子: {args.seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(os.getcwd(), f"qwen_eval_partial_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=60)

    metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "errors": 0}
    true_label_dist = {"0": 0, "1": 0}
    pred_label_dist = {"0": 0, "1": 0, "unknown": 0}

    total_chars = 0
    total_items = 0

    pred_path = os.path.join(output_dir, "predictions.csv")
    err_path = os.path.join(output_dir, "errors.txt")

    rpm = args.requests_per_minute
    min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
    last_call = 0.0

    def process_split(path: str, true_label: int, split_name: str, seed_offset: int) -> None:
        nonlocal total_chars, total_items, last_call
        print_stage(f"采样数据: {split_name.upper()} 类")
        samples = reservoir_sample_csv(path, args.text_col, args.sample_size, args.seed + seed_offset)
        print(f"  已采样 {len(samples)} 条记录")
        
        print_stage(f"模型推理: {split_name.upper()} 类")
        with open(pred_path, "a", encoding="utf-8", newline="") as pf, open(err_path, "a", encoding="utf-8") as ef:
            writer = csv.writer(pf)
            if pf.tell() == 0:
                writer.writerow(["split", "true_label", "pred_label", "raw_response", "text"])
            
            for idx, text in enumerate(tqdm(samples, desc=f"  {split_name.upper()} 推理", unit="条"), start=1):
                prompt = PROMPT_PREFIX + text
                total_chars += len(prompt)
                total_items += 1

                if min_interval > 0:
                    now = time.time()
                    elapsed = now - last_call
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                pred, raw = call_model(client, args.model, prompt)
                last_call = time.time()

                if pred is None:
                    pred_label_dist["unknown"] += 1
                    ef.write(f"{split_name}\t{idx}\t{raw}\n")
                else:
                    pred_label_dist[str(pred)] += 1

                true_label_dist[str(true_label)] += 1
                update_metrics(metrics, true_label, pred)

                writer.writerow([split_name, true_label, pred if pred is not None else "", raw, text])

    process_split(args.true_csv, 1, "true", seed_offset=1)
    process_split(args.fake_csv, 0, "fake", seed_offset=2)

    print_stage("计算评估指标")
    report = compute_report(metrics)
    report["model"] = args.model
    report["label_distribution"] = {
        "true_labels": true_label_dist,
        "pred_labels": pred_label_dist,
    }
    report["token_estimate"] = {
        "prompt_prefix_chars": len(PROMPT_PREFIX),
        "total_prompt_chars": total_chars,
        "estimated_tokens": estimate_tokens_from_chars(total_chars),
        "items": total_items,
        "sample_size_each": args.sample_size,
    }
    
    # 打印评估结果
    om = report["overall_metrics"]
    print(f"  Accuracy:  {om['accuracy']:.4f}")
    print(f"  Precision: {om['macro_precision']:.4f} (macro)")
    print(f"  Recall:    {om['macro_recall']:.4f} (macro)")
    print(f"  F1:        {om['macro_f1']:.4f} (macro)")

    print_stage("保存结果")
    report_path = os.path.join(output_dir, "metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {report_path}")

    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# {args.model} 真假判别结果（各{args.sample_size}条随机采样）\n\n")
        f.write(f"**使用模型**: `{args.model}`\n\n")
        f.write(f"- 总样本: {report['total_evaluated']} (错误: {report['errors']})\n\n")
        
        f.write("## 整体评估指标\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| Accuracy | {om['accuracy']:.4f} |\n")
        f.write(f"| Precision (macro) | {om['macro_precision']:.4f} |\n")
        f.write(f"| Recall (macro) | {om['macro_recall']:.4f} |\n")
        f.write(f"| F1 (macro) | {om['macro_f1']:.4f} |\n\n")
        
        f.write("## 各类别指标\n\n")
        f.write("| 类别 | Precision | Recall | F1 | Support |\n")
        f.write("|------|-----------|--------|----|---------|\n")
        for k, v in report["per_class"].items():
            f.write(f"| {k} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |\n")
        
        f.write("\n## 混淆矩阵\n\n")
        cm = report["confusion_matrix"]
        f.write("```\n")
        f.write("              Predicted\n")
        f.write("              0(Fake)  1(True)\n")
        f.write(f"Actual 0(Fake)  {cm['tn']:5d}    {cm['fp']:5d}\n")
        f.write(f"       1(True)  {cm['fn']:5d}    {cm['tp']:5d}\n")
        f.write("```\n\n")
        f.write(f"- TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}\n")
        
        f.write("\n## 标签分布\n\n")
        f.write(f"- 真实标签: {report['label_distribution']['true_labels']}\n")
        f.write(f"- 预测标签: {report['label_distribution']['pred_labels']}\n")
        
        f.write("\n## Token 预估\n\n")
        te = report["token_estimate"]
        f.write(f"- Prompt 前缀长度: {te['prompt_prefix_chars']} chars\n")
        f.write(f"- 总 Prompt 字符数: {te['total_prompt_chars']}\n")
        f.write(f"- 估算 tokens: {te['estimated_tokens']}\n")
    print(f"  已保存: {summary_path}")

    print_stage("完成")
    print(f"  所有结果已保存到: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
