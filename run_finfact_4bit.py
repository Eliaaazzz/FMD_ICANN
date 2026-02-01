#!/usr/bin/env python3
"""
FinFact Evaluation using BitsAndBytes 4-bit Quantization
Compatible with RTX 4060 (8GB VRAM)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os
from tqdm import tqdm
import re
import platform

# ================= 配置部分 =================
MODEL_ID = "/home/ufb/models/FMDLlama3"  # 本地微调模型 (与FinGuard一致)

# 自动检测路径 (Windows vs WSL)
if platform.system() == "Windows":
    DATA_PATH = r"C:\Users\Aufb\Desktop\FMD\processed_data\finfact\test.jsonl"
    OUTPUT_PATH = r"C:\Users\Aufb\Desktop\FMD\eval_finfact_4bit_results.json"
else:
    DATA_PATH = "/mnt/c/Users/Aufb/Desktop/FMD/processed_data/finfact/test.jsonl"
    OUTPUT_PATH = "/mnt/c/Users/Aufb/Desktop/FMD/eval_finfact_4bit_results.json"

# 测试样本数 (设为 None 跑全部1347条)
NUM_SAMPLES = None  # 全部测试
# ===========================================

def load_model():
    print(f"Loading model: {MODEL_ID} with 4-bit quantization...")

    # BitsAndBytes 4-bit 配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory Used: {allocated:.2f} GB")

    return tokenizer, model

def format_prompt(claim, evidence):
    """构造 Llama 3.1 的推理 Prompt"""
    instruction = """Task: Please determine whether the claim is True, False, or Not Enough Information (NEI) based on the contextual information provided.
The answer needs to use the following format:
Prediction: [True, or False, or NEI]
Explanation: [Brief explanation]"""

    user_content = f"{instruction}\n\nClaim: {claim}"
    if evidence:
        user_content += f"\nContextual information: {evidence}"

    # Llama 3.1 格式
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial misinformation detection expert.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def extract_prediction(text):
    """提取 True/False/NEI"""
    text_lower = text.lower()

    # 找 "Prediction: X" 模式
    match = re.search(r"prediction[:\s]*(true|false|nei)", text_lower)
    if match:
        pred = match.group(1)
        if pred == 'true': return 'True'
        if pred == 'false': return 'False'
        if pred == 'nei': return 'NEI'

    # 检查关键词
    if 'prediction: true' in text_lower or text_lower.startswith('true'):
        return 'True'
    if 'prediction: false' in text_lower or text_lower.startswith('false'):
        return 'False'
    if 'nei' in text_lower or 'not enough information' in text_lower:
        return 'NEI'

    # 简单匹配
    if 'true' in text_lower and 'false' not in text_lower:
        return 'True'
    if 'false' in text_lower and 'true' not in text_lower:
        return 'False'

    return 'Unknown'

def main():
    print("=" * 60)
    print("FinFact Evaluation - 4-bit Quantization")
    print("=" * 60)

    # 1. 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run 'prepare_finfact_data.py' first.")
        return

    # 2. 加载模型
    tokenizer, model = load_model()

    # 3. 读取数据
    print(f"\nLoading test data from {DATA_PATH}...")
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    test_data = data[:NUM_SAMPLES] if NUM_SAMPLES else data
    print(f"Running inference on {len(test_data)} samples...")

    # 统计
    correct = 0
    total = 0
    results = []
    label_correct = {'True': 0, 'False': 0, 'NEI': 0}
    label_total = {'True': 0, 'False': 0, 'NEI': 0}

    # 4. 推理循环
    for item in tqdm(test_data):
        label = item['label']
        claim = item['claim']
        evidence = item.get('evidence', '')

        prompt = format_prompt(claim, evidence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_prediction(generated_text)

        is_correct = (pred == label)
        if is_correct:
            correct += 1
            if label in label_correct:
                label_correct[label] += 1

        if label in label_total:
            label_total[label] += 1
        total += 1

        results.append({
            'claim': claim[:100],
            'label': label,
            'pred': pred,
            'correct': is_correct,
            'response': generated_text[:200]
        })

        # 打印进度
        if total % 10 == 0:
            print(f"\n[Sample {total}] Label: {label} | Pred: {pred} | {'✅' if is_correct else '❌'}")

    # 5. 结果统计
    accuracy = correct / total * 100 if total > 0 else 0

    # 计算排除NEI后的准确率 (因为模型只训练了True/False)
    non_nei_correct = label_correct['True'] + label_correct['False']
    non_nei_total = label_total['True'] + label_total['False']
    non_nei_accuracy = non_nei_correct / non_nei_total * 100 if non_nei_total > 0 else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Accuracy (all): {accuracy:.2f}% ({correct}/{total})")
    print(f"Accuracy (True/False only): {non_nei_accuracy:.2f}% ({non_nei_correct}/{non_nei_total})")
    print(f"  [Note: Model trained on FinGuard (True/False), not NEI]")
    print(f"\nPer-class accuracy:")
    for label in ['True', 'False', 'NEI']:
        if label_total[label] > 0:
            acc = label_correct[label] / label_total[label] * 100
            print(f"  {label}: {acc:.2f}% ({label_correct[label]}/{label_total[label]})")

    # 保存结果
    output = {
        'model': MODEL_ID,
        'quantization': '4-bit BitsAndBytes',
        'num_samples': total,
        'accuracy': accuracy,
        'per_class': {k: label_correct[k]/label_total[k]*100 if label_total[k] > 0 else 0 for k in label_total},
        'samples': results[:20]  # 保存前20条用于分析
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
