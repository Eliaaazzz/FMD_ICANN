import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import re

# ================= 配置部分 =================
# 显存优化版：使用 MaziyarPanahi 的 GPTQ 4-bit 模型
# 使用 TheBloke 的 GPTQ 模型 (更稳定)
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GPTQ"

# 自动检测路径 (Windows vs WSL)
import platform
if platform.system() == "Windows":
    DATA_PATH = r"C:\Users\Aufb\Desktop\FMD\processed_data\finfact\test.jsonl"
else:
    DATA_PATH = "/mnt/c/Users/Aufb/Desktop/FMD/processed_data/finfact/test.jsonl"
# ===========================================

def load_model():
    print(f"Loading model: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # device_map="auto" 会自动把模型塞进你的 4060
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: Make sure you installed auto-gptq: `pip install auto-gptq optimum`")
        exit(1)

def format_prompt(claim, evidence):
    """
    构造 Llama 3 的推理 Prompt。
    注意：这里不能包含 'Prediction:' 的答案，要让模型自己生成。
    """
    system_prompt = "You are a financial misinformation detection expert. Analyze claims and determine their veracity."
    user_content = f"Claim: {claim}\n"
    if evidence:
        user_content += f"Contextual information: {evidence}"

    # Llama 3 格式
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def extract_prediction(text):
    """简单的后处理，提取 True/False/NEI"""
    # 找 "Prediction: True" 这种模式
    match = re.search(r"Prediction:\s*(True|False|NEI)", text, re.IGNORECASE)
    if match:
        return match.group(1).title() # 转成 True/False/NEI 标准格式

    # 如果没找到标准格式，尝试找关键词
    text_lower = text.lower()
    if "prediction: true" in text_lower or "label: true" in text_lower: return "True"
    if "prediction: false" in text_lower or "label: false" in text_lower: return "False"
    if "nei" in text_lower or "not enough information" in text_lower: return "NEI"

    return "Unknown"

def main():
    # 1. 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run 'prepare_finfact_data.py' first.")
        return

    # 2. 加载模型
    tokenizer, model = load_model()
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 读取数据
    print(f"Loading test data from {DATA_PATH}...")
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # 为了快速测试，先只跑前 50 条 (想跑全部就把 [:50] 去掉)
    test_data = data[:50]
    print(f"Running inference on first {len(test_data)} samples...")

    correct = 0
    total = 0

    # 4. 推理循环
    for item in tqdm(test_data):
        label = item['label']
        claim = item['claim']
        evidence = item.get('evidence', '')

        prompt = format_prompt(claim, evidence)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128, # 不需要太长，只要 Prediction 和 Explanation
                temperature=0.1,    # 低温度保证结果稳定
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码 (只看生成的 new tokens)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 提取结果
        pred = extract_prediction(generated_text)

        # 简单比对
        is_correct = (pred == label)
        if is_correct:
            correct += 1
        total += 1

        # 打印偶尔的样例
        if total % 10 == 0:
            print(f"\n[Sample {total}]")
            print(f"Claim: {claim[:50]}...")
            print(f"Label: {label} | Pred: {pred}")
            print(f"Result: {'✅' if is_correct else '❌'}")

    # 5. 最终统计
    accuracy = correct / total * 100
    print("\n" + "="*30)
    print(f"Evaluation Complete!")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*30)

if __name__ == "__main__":
    main()
