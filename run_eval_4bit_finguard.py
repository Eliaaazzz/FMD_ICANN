#!/usr/bin/env python3
"""FMD Evaluation Script - 4-bit Quantization"""
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

def parse_pred(text):
    t = text.lower()
    if 'true' in t and 'false' not in t: return 'True'
    if 'false' in t and 'true' not in t: return 'False'
    if 'nei' in t: return 'NEI'
    return 'Unknown'

print('='*60)
print('FMD Evaluation - 4-bit Quantization')
print('='*60)

# Load model
print('\n[1/4] Loading model with 4-bit quantization...')
start = time.time()

cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    '/home/ufb/models/FMDLlama3',
    quantization_config=cfg,
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()
load_time = time.time() - start
print('  Loaded in %.2fs' % load_time)
print('  VRAM: %.2f GB' % (torch.cuda.memory_allocated()/1024**3))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/ufb/models/FMDLlama3')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load data
print('\n[2/4] Loading test data...')
with open('/home/ufb/FMD/data/full_data/FMD_test_full.json') as f:
    data = [json.loads(line) for line in f]
print('  Loaded %d samples' % len(data))

# Prepare prompts
print('\n[3/4] Preparing prompts...')
prompts = []
gts = []
for item in data:
    prompt = 'Human:\n%s\n\nAssistant:\n' % item.get('instruction', '')
    prompts.append(prompt)
    gts.append(item.get('label', parse_pred(item.get('output', ''))))

# Run inference
print('\n[4/4] Running inference...')
start = time.time()
preds = []

with torch.no_grad():
    for i, prompt in enumerate(tqdm(prompts, desc='Inferencing')):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
        
        with torch.cuda.amp.autocast():
            out = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
        
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        preds.append(parse_pred(text))

inference_time = time.time() - start

# Calculate metrics
valid_idx = [i for i in range(len(preds)) if preds[i] != 'Unknown' and gts[i] != 'Unknown']

print('\n' + '='*60)
print('RESULTS')
print('='*60)

if valid_idx:
    valid_preds = [preds[i] for i in valid_idx]
    valid_gts = [gts[i] for i in valid_idx]
    
    accuracy = accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_gts, valid_preds, average=None, zero_division=0, labels=['True', 'False', 'NEI']
    )
    
    print('Total samples: %d' % len(data))
    print('Valid predictions: %d (%.1f%%)' % (len(valid_idx), 100*len(valid_idx)/len(data)))
    print('-'*60)
    print('Accuracy: %.4f (%.2f%%)' % (accuracy, 100*accuracy))
    print('Macro F1: %.4f' % macro_f1)
    print('-'*60)
    print('Per-class metrics:')
    print('  True:  P=%.4f, R=%.4f, F1=%.4f' % (precision[0], recall[0], f1[0]))
    print('  False: P=%.4f, R=%.4f, F1=%.4f' % (precision[1], recall[1], f1[1]))
    print('  NEI:   P=%.4f, R=%.4f, F1=%.4f' % (precision[2], recall[2], f1[2]))
    print('-'*60)
    print('Time: %.2fs (%.3fs/sample)' % (inference_time, inference_time/len(data)))
    print('='*60)
    
    print('\nCOMPARISON WITH PAPER:')
    print('-'*60)
    print('Paper:  Acc=0.7362, F1=0.6667')
    print('Ours:   Acc=%.4f, F1=%.4f' % (accuracy, macro_f1))
    print('Diff:   Acc=%+.2f%%, F1=%+.4f' % (100*(accuracy-0.7362), macro_f1-0.6667))
    print('='*60)
    
    # Save results
    results = {
        'method': '4-bit-quantization',
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'precision': {'True': float(precision[0]), 'False': float(precision[1]), 'NEI': float(precision[2])},
        'recall': {'True': float(recall[0]), 'False': float(recall[1]), 'NEI': float(recall[2])},
        'f1_per_class': {'True': float(f1[0]), 'False': float(f1[1]), 'NEI': float(f1[2])},
        'total_samples': len(data),
        'valid_samples': len(valid_idx),
        'inference_time_seconds': inference_time,
        'load_time_seconds': load_time
    }
    
    with open('/home/ufb/FMD/eval_results_4bit.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\nResults saved to eval_results_4bit.json')
else:
    print('ERROR: No valid predictions!')
