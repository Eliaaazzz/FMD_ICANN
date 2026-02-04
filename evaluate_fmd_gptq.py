#!/usr/bin/env python3
"""
GPTQ量化模型的推理评估脚本
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import time

def parse_pred(text):
    """解析模型输出的预测结果"""
    text_lower = text.lower()
    
    # 寻找"Prediction: X"模式
    if 'prediction:' in text_lower:
        if 'true' in text_lower.split('prediction:')[1]:
            return 'True'
        elif 'false' in text_lower.split('prediction:')[1]:
            return 'False'
        elif 'nei' in text_lower.split('prediction:')[1]:
            return 'NEI'
    
    # 回退方案：检查最后部分
    last_part = text_lower[-300:]
    if 'true' in last_part and 'false' not in last_part:
        return 'True'
    elif 'false' in last_part and 'true' not in last_part:
        return 'False'
    elif 'nei' in last_part:
        return 'NEI'
    
    return 'Unknown'

print("="*60)
print("FMDLlama3 GPTQ推理评估")
print("="*60)

# 检查GPTQ模型是否存在
gptq_path = '/home/ufb/models/FMDLlama3-GPTQ'
if not os.path.exists(gptq_path):
    print(f"✗ 错误: 找不到GPTQ模型 {gptq_path}")
    print("请先运行: python quantize_to_gptq.py")
    exit(1)

# 加载模型和分词器
print("\n[1/4] 加载GPTQ量化模型...")
try:
    model = AutoGPTQForCausalLM.from_quantized(
        gptq_path,
        use_safetensors=True,
        device_map='auto',
        use_triton=True,
    )
    print("✓ 模型加载完成")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    exit(1)

print("\n[2/4] 加载测试数据...")
tokenizer = AutoTokenizer.from_pretrained('/home/ufb/models/FMDLlama3')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

with open('/home/ufb/FMD/data/full_data/FMD_test_full.json') as f:
    data = [json.loads(line) for line in f]

print(f"✓ 加载 {len(data)} 个测试样本")

# 准备提示和标签
prompts = [f'Human:\n{item["instruction"]}\n\nAssistant:\n' for item in data]
gts = []
for item in data:
    if 'label' in item and item['label'] != 'Unknown':
        gts.append(item['label'])
    else:
        gts.append(parse_pred(item.get('output', '')))

print(f"✓ 标签分布: {dict(Counter(gts))}")

# 推理
print("\n[3/4] 执行推理...")
preds = []
start_time = time.time()

with torch.no_grad():
    for i in tqdm(range(0, len(prompts), 8), desc="批推理"):
        batch = prompts[i:i+8]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
        ).to('cuda')
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        for j, tokens in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            response = tokenizer.decode(
                tokens[input_len:],
                skip_special_tokens=True,
            )
            preds.append(parse_pred(response))

inference_time = time.time() - start_time
print(f"✓ 推理完成 ({inference_time:.1f}秒, {len(data)/inference_time:.2f} samples/sec)")

# 计算指标
print("\n[4/4] 计算评估指标...")
valid_idx = [i for i in range(len(preds)) if preds[i] != 'Unknown' and gts[i] != 'Unknown']
valid_preds = [preds[i] for i in valid_idx]
valid_gts = [gts[i] for i in valid_idx]

if valid_preds:
    accuracy = accuracy_score(valid_gts, valid_preds)
    labels = ['True', 'False', 'NEI']
    present_labels = [l for l in labels if l in valid_preds or l in valid_gts]
    macro_f1 = f1_score(valid_gts, valid_preds, labels=present_labels, average='macro', zero_division=0)
    
    print("\n" + "="*60)
    print("推理结果")
    print("="*60)
    print(f"总样本数: {len(data)}")
    print(f"有效预测: {len(valid_idx)}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"宏平均F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"推理耗时: {inference_time:.1f}秒 (~{inference_time/60:.1f}分钟)")
    print("\n预测分布:")
    print(f"  {dict(Counter(valid_preds))}")
    print("\n分类报告:")
    print(classification_report(valid_gts, valid_preds, labels=present_labels, zero_division=0))
    print("="*60)
    
    # 保存结果
    results = {
        'method': 'GPTQ-4bit',
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'total_samples': len(data),
        'evaluated_samples': len(valid_idx),
        'inference_time_seconds': float(inference_time),
        'samples_per_second': float(len(data) / inference_time),
    }
    
    with open('/home/ufb/FMD/eval_results_gptq.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ 结果已保存到 eval_results_gptq.json")
    
    # 对比论文结果
    paper_acc = 0.7362
    paper_f1 = 0.6667
    
    print("\n与论文结果对比:")
    print(f"准确率: {accuracy:.4f} vs 论文: {paper_acc:.4f} (差异: {abs(accuracy-paper_acc)*100:.2f}%)")
    print(f"F1分数: {macro_f1:.4f} vs 论文: {paper_f1:.4f} (差异: {abs(macro_f1-paper_f1)*100:.2f}%)")

else:
    print("✗ 没有有效的预测!")
