#!/usr/bin/env python3
"""
Quick evaluation on subset with direct CPU offloading
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def parse_pred(text):
    text_lower = text.lower()
    if 'true' in text_lower and 'false' not in text_lower:
        return 'True'
    elif 'false' in text_lower and 'true' not in text_lower:
        return 'False'
    elif 'nei' in text_lower:
        return 'NEI'
    return 'Unknown'

print('Loading data...')
with open('/home/ufb/FMD/data/full_data/FMD_test_full.json') as f:
    data = [json.loads(line) for line in f]

print(f'Total: {len(data)}, using subset of {min(100, len(data))} for quick eval')
data = data[:100]  # Quick test on 100 samples

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('/home/ufb/models/FMDLlama3')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Loading model (FP32, device_map=cpu)...')
model = AutoModelForCausalLM.from_pretrained(
    '/home/ufb/models/FMDLlama3',
    device_map='cpu',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print('Model loaded on CPU')

prompts = [f'Human:\n{item["instruction"]}\n\nAssistant:\n' for item in data]
gts = []
for item in data:
    if 'label' in item and item['label'] != 'Unknown':
        gts.append(item['label'])
    else:
        gts.append(parse_pred(item.get('output', '')))

preds = []
print(f'Running inference on CPU (1 sample at a time)...')

with torch.no_grad():
    for i, prompt in enumerate(tqdm(prompts)):
        inputs = tokenizer([prompt], return_tensors='pt').to('cpu')
        
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            preds.append(parse_pred(response))
        except Exception as e:
            print(f'Error on sample {i}: {e}')
            preds.append('Unknown')

valid_idx = [i for i in range(len(preds)) if preds[i] != 'Unknown' and gts[i] != 'Unknown']
if valid_idx:
    valid_preds = [preds[i] for i in valid_idx]
    valid_gts = [gts[i] for i in valid_idx]
    
    acc = accuracy_score(valid_gts, valid_preds)
    f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)
    
    print(f'\n' + '='*50)
    print(f'RESULTS (CPU, subset={len(data)})')
    print(f'='*50)
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {f1:.4f}')
    print(f'Valid samples: {len(valid_idx)}/{len(data)}')
    print(f'='*50)
    
    with open('/home/ufb/FMD/eval_results_subset_cpu.json', 'w') as f:
        json.dump({
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'samples': len(valid_idx),
            'total': len(data),
            'note': 'CPU inference, subset of 100 samples'
        }, f, indent=2)
    print('Saved to eval_results_subset_cpu.json')
else:
    print('No valid predictions!')
