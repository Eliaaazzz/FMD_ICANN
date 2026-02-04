#!/usr/bin/env python3
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

print(f'Total samples: {len(data)}')
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('/home/ufb/models/FMDLlama3')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

print('Loading model in fp16 (no quantization)...')
model = AutoModelForCausalLM.from_pretrained(
    '/home/ufb/models/FMDLlama3',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()
print(f'Model loaded! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB')

prompts = [f'Human:\n{item["instruction"]}\n\nAssistant:\n' for item in data]
gts = []
for item in data:
    if 'label' in item and item['label'] != 'Unknown':
        gts.append(item['label'])
    else:
        gts.append(parse_pred(item.get('output', '')))

preds = []
print(f'Running inference with batch_size=2 (fp16, no quantization)...')
with torch.no_grad():
    for i in tqdm(range(0, len(prompts), 2), desc='Inference'):
        batch = prompts[i:i+2]
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

valid_idx = [i for i in range(len(preds)) if preds[i] != 'Unknown' and gts[i] != 'Unknown']
print(f'\nEvaluating {len(valid_idx)} samples with valid predictions...')

valid_preds = [preds[i] for i in valid_idx]
valid_gts = [gts[i] for i in valid_idx]

acc = accuracy_score(valid_gts, valid_preds)
f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)

print(f'\n' + '='*50)
print(f'RESULTS (FP16, No Quantization)')
print(f'='*50)
print(f'Accuracy: {acc:.4f} ({acc*100:.2f}%)')
print(f'Macro F1: {f1:.4f} ({f1*100:.2f}%)')
print(f'Evaluated Samples: {len(valid_idx)}')
print(f'='*50)

results = {
    'accuracy': float(acc),
    'macro_f1': float(f1),
    'total_samples': len(data),
    'evaluated_samples': len(valid_idx),
    'note': 'FP16 without quantization'
}

with open('/home/ufb/FMD/eval_results_fp16.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Results saved to eval_results_fp16.json')
