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
DATA_PATH = "data/FinFact/finfact_100.json"
OUTPUT_DIR = "FinFact"
SLEEP_INTERVAL = 0.1  # API 调用间隔（秒）

# ============================================================
# 初始化 Qwen 客户端
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 定义多个不同的 Prompt 模板
# 使用字段说明:
# - claim: 待验证的声明
# - sci_digest: 声明的核心摘要
# - justification: 事实核查的完整论证文本
# - evidence: 支撑论证的可溯源依据
# ============================================================
PROMPT_TEMPLATES = {
    "simple": {
        "system": "你是一个事实核查助手。只输出判断结果，不需要解释。",
        "user": """请判断以下声明的真实性。

## 待验证声明
{claim}

## 声明摘要
{sci_digest}

## 事实核查论证
{justification}

只需回答 True（声明属实）或 False（声明虚假/误导），格式：Prediction: True 或 Prediction: False"""
    },
    
    "detailed_criteria": {
        "system": """你是一位经验丰富的事实核查专家，专门识别虚假信息和误导性内容。
你具备以下专业能力：
1. 识别煽动性、情绪化的语言模式
2. 判断信息来源的可靠性和权威性
3. 检测逻辑谬误和事实不一致
4. 识别政治偏见和议程驱动的内容""",
        "user": """请运用你的专业知识，对以下声明进行全面核查。

【待核查声明】
{claim}

【声明核心摘要】
{sci_digest}

【事实核查论证】
{justification}

【支撑证据】
{evidence}

【核查维度】
1. 声明准确性：声明是否准确反映了事实？是否存在断章取义或曲解？
2. 证据充分性：提供的证据是否足以支持或反驳该声明？
3. 逻辑一致性：论证过程是否自洽？是否存在逻辑漏洞？
4. 来源可信度：证据来源是否可靠？引用是否准确？
5. 误导性检测：声明是否具有误导公众的倾向？

基于以上维度的综合分析，给出最终判断。
输出格式：Prediction: True 或 Prediction: False"""
    },
    
    "cot_stepwise": {
        "system": """你是一个采用链式思维(Chain-of-Thought)方法的事实核查AI。
你会按照结构化的步骤进行分析，确保判断的严谨性和可追溯性。""",
        "user": """请使用链式思维方法，逐步分析以下声明的真实性。

【声明内容】
{claim}

【声明摘要】
{sci_digest}

【事实核查论证】
{justification}

【支撑证据】
{evidence}

【分析步骤】
Step 1 - 声明解析：这个声明的核心主张是什么？摘要是否准确概括了关键点？
Step 2 - 证据评估：提供的证据是否充分？来源是否可靠？
Step 3 - 论证分析：事实核查论证是否支持或反驳该声明？
Step 4 - 语境检验：声明是否被正确理解？是否存在断章取义？
Step 5 - 综合判断：基于以上分析，该声明是否属实？

完成分析后，输出最终判断：Prediction: True 或 Prediction: False"""
    },
    
    "financial_expert": {
        "system": """你是一位拥有20年经验的资深金融分析师和事实核查专家。
你曾在华尔街日报、彭博社等权威金融媒体工作，对金融相关声明的真实性判断有敏锐的洞察力。
你特别擅长识别：市场操纵性假新闻、投资诈骗宣传、夸大的财务数据、政策误读。""",
        "user": """作为资深金融事实核查专家，请对以下声明进行专业评估。

【待评估声明】
{claim}

【声明核心摘要】
{sci_digest}

【详细论证分析】
{justification}

【可溯源证据】
{evidence}

【专业核查要点】
1. 事实准确性：声明中涉及的事实、数据是否准确？
2. 语境完整性：声明是否在正确的语境下被理解？
3. 证据可靠性：引用的证据来源是否权威可信？
4. 专业解读：从金融/经济专业角度，该声明是否存在误导？
5. 影响评估：该声明是否可能误导公众做出错误判断？

请给出你的专业判断：Prediction: True 或 Prediction: False"""
    },
    
    "binary_classifier_en": {
        "system": """You are a sophisticated binary classifier specialized in fact-checking claims.
Your classification is based on evidence analysis, logical consistency, and source credibility.
You have been trained on millions of verified fact-check cases.""",
        "user": """Analyze the following claim and classify it as true or false based on the provided evidence.

[CLAIM]
{claim}

[CLAIM SUMMARY]
{sci_digest}

[FACT-CHECK ANALYSIS]
{justification}

[SUPPORTING EVIDENCE]
{evidence}

[CLASSIFICATION CRITERIA]
- Accuracy: Does the claim accurately represent the facts?
- Context: Is the claim presented in proper context?
- Evidence: Does the evidence support or refute the claim?
- Sources: Are the cited sources credible and verifiable?
- Misleading: Is the claim potentially misleading?

Provide your binary classification.
Output format: Prediction: True (claim is accurate) or Prediction: False (claim is false/misleading)"""
    },
    
    "skeptical_investigator": {
        "system": """你是一位极度怀疑的调查记者，在揭露虚假声明方面有着丰富的经验。
你的座右铭是："非经验证，皆为可疑"。
你会从最严格的标准审视每一个声明，寻找任何可能的漏洞和不实之处。""",
        "user": """以调查记者的严格标准，审查以下声明的真实性。

【待审查声明】
{claim}

【声明摘要】
{sci_digest}

【核查论证材料】
{justification}

【可溯源证据清单】
{evidence}

【调查审查清单】
□ 事实核实：声明中的每个事实陈述是否都有证据支持？
□ 证据溯源：提供的证据来源是否可追溯、可验证？
□ 引用准确性：原话或原意是否被正确引用？
□ 语境完整性：声明是否在完整语境下呈现？
□ 逻辑严密性：从证据到结论的推理是否严密？
□ 误导可能性：声明是否存在误导公众的风险？

基于严格审查，给出判断：Prediction: True 或 Prediction: False"""
    },
    
    "concise": {
        "system": "事实核查二分类器。直接输出分类结果。",
        "user": """声明：{claim}
摘要：{sci_digest}
论证：{justification}

分类结果：Prediction: True 或 Prediction: False"""
    },
    
    "multi_perspective": {
        "system": """你是一个多角度分析系统，会从不同视角审视声明的真实性。
你会考虑：事实核查员视角、领域专家视角、普通读者视角、批判性思维视角。""",
        "user": """请从多个角度分析以下声明的真实性。

【声明内容】
{claim}

【声明摘要】
{sci_digest}

【事实核查论证】
{justification}

【支撑证据】
{evidence}

【多角度分析框架】
🔍 事实核查员视角：声明的核心事实是否可验证？证据是否充分？来源是否可靠？
🎓 领域专家视角：从专业角度看，声明是否准确？术语使用是否正确？
👤 普通读者视角：声明是否可能误导不了解背景的读者？
🧠 批判性思维视角：论证逻辑是否严密？是否存在逻辑谬误？证据链是否完整？

综合以上视角，输出判断：Prediction: True 或 Prediction: False"""
    }
}



def format_evidence(evidence_list: list[dict]) -> str:
    """格式化证据列表为字符串
    
    每条证据包含:
    - sentence: 证据描述
    - hrefs: 来源链接列表
    """
    if not evidence_list:
        return "无可用证据"
    
    # 确保是列表类型
    if not isinstance(evidence_list, list):
        return "无可用证据"

    formatted = []
    for i, ev in enumerate(evidence_list, 1):
        # 确保ev是字典类型
        if not isinstance(ev, dict):
            continue
            
        sentence = ev.get("sentence", "") or ""
        hrefs = ev.get("hrefs", []) or []
        
        # 过滤掉None值并确保都是字符串
        valid_hrefs = [str(h) for h in hrefs if h is not None]
        
        href_str = ", ".join(valid_hrefs[:3]) if valid_hrefs else "无来源链接"
        if valid_hrefs and len(valid_hrefs) > 3:
            href_str += f" (等{len(valid_hrefs)}个来源)"
        
        if sentence:  # 只添加有内容的证据
            formatted.append(f"  证据{i}: {sentence}\n    来源: {href_str}")

    return "\n".join(formatted) if formatted else "无可用证据"



def load_data() -> list[dict]:
    """加载FinFact JSON数据
    
    使用字段:
    - claim: 待验证的声明
    - sci_digest: 声明的核心摘要（列表）
    - justification: 事实核查的完整论证文本
    - evidence: 支撑论证的可溯源依据（列表）
    - label: 标签 (true/false)
    """
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 尝试修复不完整的JSON（文件可能被截断）
    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("尝试修复截断的JSON...")
        
        # 找到最后一个完整的对象（以 "}," 或 "}" 结尾）
        # 方法1: 尝试找到最后一个完整的 "},\n  {" 或 "}\n]" 模式
        last_complete = content.rfind('},\n  {')
        if last_complete == -1:
            last_complete = content.rfind('},\n{')
        if last_complete == -1:
            last_complete = content.rfind('}\n]')
        
        if last_complete != -1:
            # 截取到最后一个完整对象，并补上 "]"
            fixed_content = content[:last_complete + 1] + "\n]"
            try:
                raw_data = json.loads(fixed_content)
                print(f"修复成功！加载了 {len(raw_data)} 条记录")
            except json.JSONDecodeError:
                # 方法2: 逐行尝试找到有效的JSON结尾
                lines = content.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == '},':
                        # 去掉最后的逗号，加上 "]"
                        test_content = '\n'.join(lines[:i+1])[:-1] + '\n]'
                        try:
                            raw_data = json.loads(test_content)
                            print(f"修复成功！加载了 {len(raw_data)} 条记录")
                            break
                        except:
                            continue
                    elif lines[i].strip() == '}':
                        test_content = '\n'.join(lines[:i+1]) + '\n]'
                        try:
                            raw_data = json.loads(test_content)
                            print(f"修复成功！加载了 {len(raw_data)} 条记录")
                            break
                        except:
                            continue
                else:
                    raise ValueError("无法修复JSON文件，请检查数据文件格式")
        else:
            raise ValueError("无法修复JSON文件，请检查数据文件格式")
    
    data = []
    for item in raw_data:
        label = item.get("label", "").strip().lower()
        # 只处理 true 和 false 标签，跳过其他（如 NEI）
        if label in ["true", "false"]:
            # 处理 sci_digest（列表转字符串）
            sci_digest = item.get("sci_digest", [])
            if isinstance(sci_digest, list):
                sci_digest_text = " ".join(sci_digest)
            else:
                sci_digest_text = str(sci_digest) if sci_digest else ""
            
            # 处理 evidence（格式化为字符串）
            evidence = item.get("evidence", [])
            evidence_text = format_evidence(evidence)
            
            data.append({
                "claim": item.get("claim", ""),
                "sci_digest": sci_digest_text,
                "justification": item.get("justification", ""),
                "evidence": evidence_text,
                "label": "True" if label == "true" else "False",
                "url": item.get("url", ""),
                "author": item.get("author", "")
            })
    
    # 根据 NUM_SAMPLES 限制样本数
    if NUM_SAMPLES > 0:
        true_samples = [d for d in data if d["label"] == "True"][:NUM_SAMPLES]
        false_samples = [d for d in data if d["label"] == "False"][:NUM_SAMPLES]
        data = true_samples + false_samples
    
    return data


def build_instruction(item: dict, prompt_template: dict) -> tuple[str, str]:
    """根据模板构建指令
    
    使用所有可用字段:
    - claim: 待验证的声明
    - sci_digest: 声明的核心摘要
    - justification: 事实核查的完整论证文本
    - evidence: 支撑论证的可溯源依据
    """
    system_prompt = prompt_template["system"]
    user_prompt = prompt_template["user"].format(
        claim=item["claim"],
        sci_digest=item["sci_digest"][:500] if item["sci_digest"] else "无摘要",
        justification=item["justification"][:2500] if item["justification"] else "无论证",
        evidence=item["evidence"][:1000] if item["evidence"] else "无证据"
    )
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


def run_experiment_with_prompt(data: list[dict], prompt_name: str, prompt_template: dict) -> dict:
    """使用指定的prompt模板运行实验"""
    print(f"\n{'='*60}")
    print(f"测试 Prompt: {prompt_name}")
    print(f"{'='*60}")
    
    y_true = []
    y_pred = []
    details = []
    
    for i, item in enumerate(data):
        system_prompt, user_prompt = build_instruction(item, prompt_template)
        ground_truth = item["label"]
        
        print(f"  处理第 {i + 1}/{len(data)} 条...", end=" ")
        print(f"[{item['claim'][:40]}...]", end=" ")
        
        response = call_qwen_api(system_prompt, user_prompt)
        prediction = extract_prediction(response)
        
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth}, 预测: {prediction} {is_correct}")
        
        details.append({
            "index": i,
            "claim": item["claim"],
            "url": item["url"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": ground_truth == prediction
        })
        
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
                "claim": detail["claim"],
                "url": detail["url"],
                "ground_truth": detail["ground_truth"],
                "prediction": detail["prediction"],
                "correct": detail["correct"]
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
    
    print(f"{'Prompt名称':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    
    for result in sorted_results:
        m = result["metrics"]
        print(f"{result['prompt_name']:<25} {m['accuracy']:>10.2%} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%}")
    
    # 找出最佳prompt
    best = sorted_results[0]
    print(f"\n🏆 最佳 Prompt: {best['prompt_name']} (F1: {best['metrics']['f1']:.2%})")


def main():
    print("=" * 60)
    print("FinFact 数据集 - 多Prompt性能测试实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"测试Prompt数量: {len(PROMPT_TEMPLATES)}")
    print()
    
    # 加载数据
    print("[1/3] 加载数据...")
    data = load_data()
    print(f"      总数据量: {len(data)} 条")
    
    label_dist = Counter([item["label"] for item in data])
    print(f"      标签分布: {dict(label_dist)}")
    
    # 运行所有prompt实验
    print("\n[2/3] 开始多Prompt测试...")
    all_results = []
    
    for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
        result = run_experiment_with_prompt(data, prompt_name, prompt_template)
        all_results.append(result)
    
    # 保存结果
    print("\n[3/3] 保存结果...")
    save_results(all_results, OUTPUT_DIR)
    
    # 打印最终比较
    print_final_comparison(all_results)
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
