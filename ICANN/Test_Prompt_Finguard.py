import json
import re
import time
import pandas as pd
from datetime import datetime
from openai import OpenAI
from collections import Counter
import os

# ============================================================
# 超参数设置
# ============================================================
NUM_SAMPLES = -1  # 【可调】每类测试样本数，设为 -1 表示全部测试
MODEL_NAME = "qwen3-max"
API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"
TRUE_DATA_PATH = "data/FinGuard/Finance_TRUE_50.csv"
FAKE_DATA_PATH = "data/FinGuard/Finance_FAKE_50.csv"
OUTPUT_DIR = "FinGuard"
SLEEP_INTERVAL = 0.1  # API 调用间隔（秒）
CHECKPOINT_FILE = "FinGuard/checkpoint_finguard.json"  # 断点续跑进度文件

# ============================================================
# 初始化 Qwen 客户端
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 定义多个不同的 Prompt 模板
# ============================================================
PROMPT_TEMPLATES = {
#     "simple": {
#         "system": "你是一个新闻真实性判断助手。只输出判断结果，不需要解释。",
#         "user": """请判断以下新闻的真实性。

# 新闻内容：
# {text}

# 只需回答 True（真实新闻）或 False（虚假新闻），格式：Prediction: True 或 Prediction: False"""
#     },
    
#     "detailed_criteria": {
#         "system": """你是一位经验丰富的新闻事实核查专家，专门识别虚假信息和误导性内容。
# 你具备以下专业能力：
# 1. 识别煽动性、情绪化的语言模式
# 2. 判断信息来源的可靠性和权威性
# 3. 检测逻辑谬误和事实不一致
# 4. 识别政治偏见和议程驱动的内容""",
#         "user": """请运用你的专业知识，对以下新闻进行全面核查。

# 【待核查新闻】
# {text}

# 【核查维度】
# 1. 语言分析：是否存在夸大、煽动性、情绪化表达？是否使用了"惊人"、"震惊"等吸引眼球的词汇？
# 2. 来源可信度：新闻是否引用了可验证的来源？引用的专家或机构是否真实存在？
# 3. 逻辑一致性：报道内容是否自洽？是否存在前后矛盾？
# 4. 事实核查：涉及的数据、日期、人物等是否准确？
# 5. 偏见检测：是否存在明显的政治立场或意识形态倾向？

# 基于以上维度的综合分析，给出最终判断。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },
    
    "cot_stepwise": {
        "system": """你是一个采用链式思维(Chain-of-Thought)方法的新闻核查AI。
你会按照结构化的步骤进行分析，确保判断的严谨性和可追溯性。""",
        "user": """请使用链式思维方法，逐步分析以下新闻的真实性。

【新闻内容】
{text}

【分析步骤】
Step 1 - 内容摘要：这篇新闻的核心主张是什么？
Step 2 - 语言风格：使用的语言是客观中立的还是煽动性的？
Step 3 - 证据评估：文中提供了哪些支持性证据？这些证据可信吗？
Step 4 - 逻辑检验：论证过程是否合理？是否存在逻辑跳跃？
Step 5 - 综合判断：基于以上分析，得出结论。

完成分析后，输出最终判断：Prediction: True 或 Prediction: False"""
    },
    
    "financial_expert": {
        "system": """你是一位拥有20年经验的资深金融分析师和新闻核查专家。
你曾在华尔街日报、彭博社等权威金融媒体工作，对金融新闻的真实性判断有敏锐的洞察力。
你特别擅长识别：市场操纵性假新闻、投资诈骗宣传、夸大的财务数据、虚假的专家背书。""",
        "user": """作为资深金融新闻核查专家，请对以下新闻进行专业评估。

【待评估新闻】
{text}

【专业核查要点】
1. 金融数据准确性：涉及的股价、市值、财务数据是否合理？
2. 市场影响分析：该新闻是否有操纵市场情绪的意图？
3. 来源权威性：消息来源是否为公认的金融机构或监管部门？
4. 专业术语使用：金融术语的使用是否正确？是否存在误导性表述？
5. 时效性检查：新闻涉及的时间点和事件是否匹配？

请给出你的专业判断：Prediction: True 或 Prediction: False"""
    },
    
    "binary_classifier_en": {
        "system": """You are a sophisticated binary classifier specialized in detecting fake news and misinformation.
Your classification is based on linguistic patterns, factual consistency, and source credibility analysis.
You have been trained on millions of verified real and fake news articles.""",
        "user": """Analyze the following news article and classify it as authentic or fabricated.

[NEWS ARTICLE]
{text}

[CLASSIFICATION CRITERIA]
- Linguistic markers: sensationalism, emotional manipulation, clickbait patterns
- Factual indicators: verifiable claims, credible sources, logical consistency
- Structural elements: professional journalism standards, balanced reporting

Provide your binary classification.
Output format: Prediction: True (authentic) or Prediction: False (fake/misleading)"""
    },
    
    "skeptical_investigator": {
        "system": """你是一位极度怀疑的调查记者，在揭露虚假新闻方面有着丰富的经验。
你的座右铭是："非经验证，皆为可疑"。
你会从最严格的标准审视每一条新闻，寻找任何可能的漏洞和不实之处。""",
        "user": """以调查记者的严格标准，审查以下新闻的真实性。

【待审查新闻】
{text}

【调查审查清单】
□ 事实依据：报道中的每个事实陈述是否都有可验证的来源？
□ 语言中立性：是否使用了中立客观的报道语言？有无情绪煽动？
□ 逻辑完整性：论证链条是否完整？是否存在逻辑跳跃或隐藏假设？
□ 利益关联：报道是否可能服务于特定利益群体的议程？
□ 时间线验证：事件的时间顺序是否合理？是否与已知事实相符？
□ 专家引用：引用的专家是否真实存在？其言论是否被正确引用？

基于严格审查，给出判断：Prediction: True 或 Prediction: False"""
    },
    
    "concise": {
        "system": "新闻真假二分类器。直接输出分类结果。",
        "user": """新闻：{text}

分类结果：Prediction: True 或 Prediction: False"""
    },
    
    "multi_perspective": {
        "system": """你是一个多角度分析系统，会从不同视角审视新闻的真实性。
你会考虑：记者视角、事实核查员视角、普通读者视角、领域专家视角。""",
        "user": """请从多个角度分析以下新闻的真实性。

【新闻内容】
{text}

【多角度分析框架】
📰 记者视角：报道是否遵循新闻写作规范？结构是否专业？
🔍 事实核查员视角：核心事实是否可验证？数据是否准确？
👤 普通读者视角：内容是否试图激起强烈情绪反应？
🎓 领域专家视角：专业内容是否准确？术语使用是否正确？

综合以上视角，输出判断：Prediction: True 或 Prediction: False"""
    }
}


def load_data() -> list[dict]:
    """加载两个CSV数据文件，合并为统一格式"""
    data = []
    
    # 加载真实新闻
    df_true = pd.read_csv(TRUE_DATA_PATH)
    text_col = df_true.columns[0]  # 假设第一列是文本
    true_samples = df_true if NUM_SAMPLES < 0 else df_true.head(NUM_SAMPLES)
    for idx, row in true_samples.iterrows():
        data.append({
            "text": str(row[text_col]),
            "label": "True",
            "source": "Finance_TRUE_50"
        })
    
    # 加载虚假新闻
    df_fake = pd.read_csv(FAKE_DATA_PATH)
    text_col = df_fake.columns[0]
    fake_samples = df_fake if NUM_SAMPLES < 0 else df_fake.head(NUM_SAMPLES)
    for idx, row in fake_samples.iterrows():
        data.append({
            "text": str(row[text_col]),
            "label": "False",
            "source": "Finance_FAKE_50"
        })
    
    return data


def build_instruction(text: str, prompt_template: dict) -> tuple[str, str]:
    """根据模板构建指令"""
    system_prompt = prompt_template["system"]
    user_prompt = prompt_template["user"].format(text=text[:2000])  # 限制长度
    return system_prompt, user_prompt


def call_qwen_api(system_prompt: str, user_prompt: str) -> str:
    """调用 Qwen API 获取预测结果"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用出错: {e}")
        return ""


def extract_prediction(response_text: str) -> str:
    """从模型回复中提取预测标签"""
    if not response_text:
        return "Unknown"
    
    # 尝试匹配 "Prediction: True/False" 格式
    match = re.search(r"Prediction:\s*(True|False)", response_text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        return "True" if label == "true" else "False"
    
    # 备选：直接查找 True/False
    response_lower = response_text.lower()
    
    # 查找最后出现的 true 或 false
    true_pos = response_lower.rfind("true")
    false_pos = response_lower.rfind("false")
    
    if true_pos > false_pos:
        return "True"
    elif false_pos > true_pos:
        return "False"
    
    return "Unknown"


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """计算评估指标"""
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
    
    if len(valid_pairs) == 0:
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
            "total": 0, "correct": 0, "unknown_count": len(y_true)
        }
    
    correct = sum(1 for t, p in valid_pairs if t == p)
    total = len(valid_pairs)
    accuracy = correct / total if total > 0 else 0
    
    # 计算二分类指标 (以 True 为正类)
    tp = sum(1 for t, p in valid_pairs if t == "True" and p == "True")
    fp = sum(1 for t, p in valid_pairs if t == "False" and p == "True")
    fn = sum(1 for t, p in valid_pairs if t == "True" and p == "False")
    tn = sum(1 for t, p in valid_pairs if t == "False" and p == "False")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct,
        "unknown_count": len(y_true) - len(valid_pairs),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


# ============================================================
# 断点续跑功能
# ============================================================
def load_checkpoint() -> dict:
    """加载断点进度"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"✅ 发现断点文件，将从上次中断处继续...")
            print(f"   已完成的Prompt: {list(checkpoint.get('completed_prompts', {}).keys())}")
            return checkpoint
        except Exception as e:
            print(f"⚠️ 加载断点文件失败: {e}，将重新开始")
    return {"completed_prompts": {}, "current_prompt": None, "current_index": 0}


def save_checkpoint(checkpoint: dict):
    """保存断点进度"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """清除断点文件（实验完成后调用）"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🗑️ 断点文件已清除")


def run_experiment_with_prompt(data: list[dict], prompt_name: str, prompt_template: dict,
                                checkpoint: dict = None) -> dict:
    """使用指定的prompt模板运行实验（支持断点续跑）"""
    print(f"\n{'='*60}")
    print(f"测试 Prompt: {prompt_name}")
    print(f"{'='*60}")
    
    y_true = []
    y_pred = []
    details = []
    
    # 检查是否需要从断点恢复
    start_index = 0
    if checkpoint and checkpoint.get("current_prompt") == prompt_name:
        start_index = checkpoint.get("current_index", 0)
        # 恢复已完成的数据
        if "partial_results" in checkpoint:
            partial = checkpoint["partial_results"]
            y_true = partial.get("y_true", [])
            y_pred = partial.get("y_pred", [])
            details = partial.get("details", [])
        if start_index > 0:
            print(f"  📍 从第 {start_index + 1} 条继续（已完成 {start_index} 条）")
    
    for i, item in enumerate(data):
        # 跳过已处理的样本
        if i < start_index:
            continue
            
        system_prompt, user_prompt = build_instruction(item["text"], prompt_template)
        ground_truth = item["label"]
        
        print(f"  处理第 {i + 1}/{len(data)} 条...", end=" ")
        
        response = call_qwen_api(system_prompt, user_prompt)
        prediction = extract_prediction(response)
        
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth}, 预测: {prediction} {is_correct}")
        
        details.append({
            "index": i,
            "source": item["source"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": ground_truth == prediction,
            "text_preview": item["text"][:100] + "..."
        })
        
        # 每处理一条就保存进度
        if checkpoint is not None:
            checkpoint["current_prompt"] = prompt_name
            checkpoint["current_index"] = i + 1
            checkpoint["partial_results"] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "details": details
            }
            save_checkpoint(checkpoint)
        
        time.sleep(SLEEP_INTERVAL)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n【{prompt_name}】结果汇总:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1']:.2%}")
    
    return {
        "prompt_name": prompt_name,
        "system_prompt": prompt_template["system"],
        "user_prompt_template": prompt_template["user"],
        "metrics": metrics,
        "details": details,
        "y_true": y_true,
        "y_pred": y_pred
    }


def save_results(all_results: list[dict], output_dir: str):
    """保存结果到CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存性能汇总表
    summary_data = []
    for result in all_results:
        m = result["metrics"]
        summary_data.append({
            "prompt_name": result["prompt_name"],
            "accuracy": f"{m['accuracy']:.4f}",
            "precision": f"{m['precision']:.4f}",
            "recall": f"{m['recall']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "total_samples": m["total"],
            "correct": m["correct"],
            "unknown_count": m["unknown_count"],
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
            "tn": m.get("tn", 0),
            "system_prompt": result["system_prompt"],
            "user_prompt_template": result["user_prompt_template"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"prompt_comparison_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"\n性能汇总已保存至: {summary_file}")
    
    # 2. 保存详细预测结果
    detailed_data = []
    for result in all_results:
        for detail in result["details"]:
            detailed_data.append({
                "prompt_name": result["prompt_name"],
                "index": detail["index"],
                "source": detail["source"],
                "ground_truth": detail["ground_truth"],
                "prediction": detail["prediction"],
                "correct": detail["correct"],
                "text_preview": detail["text_preview"]
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = os.path.join(output_dir, f"prompt_comparison_details_{timestamp}.csv")
    detailed_df.to_csv(detailed_file, index=False, encoding="utf-8-sig")
    print(f"详细结果已保存至: {detailed_file}")
    
    # 3. 保存完整JSON（包含所有信息）
    json_file = os.path.join(output_dir, f"prompt_comparison_full_{timestamp}.json")
    json_output = []
    for result in all_results:
        json_output.append({
            "prompt_name": result["prompt_name"],
            "system_prompt": result["system_prompt"],
            "user_prompt_template": result["user_prompt_template"],
            "metrics": result["metrics"]
        })
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"完整JSON已保存至: {json_file}")
    
    return summary_file, detailed_file, json_file


def print_final_comparison(all_results: list[dict]):
    """打印最终比较结果"""
    print("\n" + "=" * 80)
    print("【最终性能比较】")
    print("=" * 80)
    
    # 按F1分数排序
    sorted_results = sorted(all_results, key=lambda x: x["metrics"]["f1"], reverse=True)
    
    print(f"{'Prompt名称':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    
    for result in sorted_results:
        m = result["metrics"]
        print(f"{result['prompt_name']:<20} {m['accuracy']:>10.2%} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%}")
    
    # 找出最佳prompt
    best = sorted_results[0]
    print(f"\n🏆 最佳 Prompt: {best['prompt_name']} (F1: {best['metrics']['f1']:.2%})")


def main():
    print("=" * 60)
    print("FinGuard 数据集 - 多Prompt性能测试实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"测试Prompt数量: {len(PROMPT_TEMPLATES)}")
    print()
    
    # 加载断点进度
    checkpoint = load_checkpoint()
    
    # 加载数据
    print("[1/3] 加载数据...")
    data = load_data()
    print(f"      总数据量: {len(data)} 条")
    
    label_dist = Counter([item["label"] for item in data])
    print(f"      标签分布: {dict(label_dist)}")
    
    # 运行所有prompt实验
    print("\n[2/3] 开始多Prompt测试...")
    all_results = []
    
    # 恢复已完成的prompt结果
    completed_prompts = checkpoint.get("completed_prompts", {})
    
    for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
        # 如果该prompt已完成，直接使用缓存结果
        if prompt_name in completed_prompts:
            print(f"\n{'='*60}")
            print(f"⏭️ 跳过已完成的 Prompt: {prompt_name}")
            print(f"{'='*60}")
            all_results.append(completed_prompts[prompt_name])
            continue
        
        # 运行实验
        result = run_experiment_with_prompt(data, prompt_name, prompt_template, checkpoint)
        all_results.append(result)
        
        # 保存该prompt的完整结果到checkpoint
        checkpoint["completed_prompts"][prompt_name] = result
        checkpoint["current_prompt"] = None  # 清除当前进度
        checkpoint["current_index"] = 0
        checkpoint.pop("partial_results", None)
        save_checkpoint(checkpoint)
        print(f"  💾 Prompt [{prompt_name}] 结果已保存到断点文件")
    
    # 保存结果
    print("\n[3/3] 保存结果...")
    save_results(all_results, OUTPUT_DIR)
    
    # 打印最终比较
    print_final_comparison(all_results)
    
    # 清除断点文件
    clear_checkpoint()
    
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
