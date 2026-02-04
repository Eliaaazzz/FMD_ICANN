#!/usr/bin/env python3
"""
vLLM-based evaluation for FMD task
Provides 4-8x speedup over standard PyTorch inference
"""

import json
import time
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from vllm import LLM, SamplingParams
import warnings

warnings.filterwarnings('ignore')

def parse_pred(text):
    """Parse model output to True/False/NEI format"""
    text_lower = text.lower()
    if 'true' in text_lower and 'false' not in text_lower:
        return 'True'
    elif 'false' in text_lower and 'true' not in text_lower:
        return 'False'
    elif 'nei' in text_lower:
        return 'NEI'
    return 'Unknown'

def main():
    print('='*60)
    print('FMD Evaluation using vLLM')
    print('='*60)
    
    # Load model with vLLM
    print('\n[1/4] Loading FMDLlama3 model with vLLM...')
    start_time = time.time()
    
    llm = LLM(
        model='/home/ufb/models/FMDLlama3',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype='float16',
        trust_remote_code=True,
        max_model_len=2048,
    )
    
    load_time = time.time() - start_time
    print(f'✓ Model loaded in {load_time:.2f}s')
    
    # Load test data
    print('\n[2/4] Loading test data...')
    with open('/home/ufb/FMD/data/full_data/FMD_test_full.json') as f:
        data = [json.loads(line) for line in f]
    
    print(f'✓ Loaded {len(data)} test samples')
    
    # Prepare prompts
    print('\n[3/4] Preparing prompts...')
    prompts = []
    gts = []
    
    for item in data:
        prompt = f'Human:\n{item["instruction"]}\n\nAssistant:\n'
        prompts.append(prompt)
        gts.append(item.get('label', parse_pred(item.get('output', ''))))
    
    print(f'✓ Prepared {len(prompts)} prompts')
    
    # Run inference
    print('\n[4/4] Running inference with vLLM...')
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        top_p=1.0,
    )
    
    start_inference = time.time()
    
    # vLLM can handle all prompts at once with automatic batching
    outputs = llm.generate(prompts, sampling_params)
    
    inference_time = time.time() - start_inference
    
    # Parse predictions
    preds = []
    for output in outputs:
        text = output.outputs[0].text
        pred = parse_pred(text)
        preds.append(pred)
    
    # Calculate metrics
    valid_idx = [i for i in range(len(preds)) 
                 if preds[i] != 'Unknown' and gts[i] != 'Unknown']
    
    if not valid_idx:
        print('ERROR: No valid predictions')
        return
    
    valid_preds = [preds[i] for i in valid_idx]
    valid_gts = [gts[i] for i in valid_idx]
    
    accuracy = accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_gts, valid_preds, average=None, zero_division=0, labels=['True', 'False', 'NEI']
    )
    
    # Print results
    print('\n' + '='*60)
    print('EVALUATION RESULTS (vLLM)')
    print('='*60)
    print(f'Total samples: {len(data)}')
    print(f'Valid predictions: {len(valid_idx)} ({len(valid_idx)/len(data)*100:.1f}%)')
    print(f'Invalid/Unknown: {len(data)-len(valid_idx)}')
    print('-'*60)
    print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Macro F1: {macro_f1:.4f}')
    print('-'*60)
    print('Per-class metrics:')
    print(f'  True:  Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}')
    print(f'  False: Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}')
    print(f'  NEI:   Precision={precision[2]:.4f}, Recall={recall[2]:.4f}, F1={f1[2]:.4f}')
    print('-'*60)
    print(f'Inference time: {inference_time:.2f}s ({inference_time/len(data):.3f}s/sample)')
    print(f'Total time: {load_time + inference_time:.2f}s')
    print('='*60)
    
    # Compare with paper baseline
    print('\nCOMPARISON WITH PAPER BASELINE:')
    print('-'*60)
    paper_accuracy = 0.7362
    paper_f1 = 0.6667
    print(f'Paper Accuracy:  {paper_accuracy:.4f} (73.62%)')
    print(f'Our Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Difference:      {(accuracy-paper_accuracy)*100:+.2f}%')
    print()
    print(f'Paper Macro F1:  {paper_f1:.4f} (66.67%)')
    print(f'Our Macro F1:    {macro_f1:.4f}')
    print(f'Difference:      {(macro_f1-paper_f1):+.4f}')
    print('='*60)
    
    # Save results
    results = {
        'method': 'vLLM',
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'precision_true': float(precision[0]),
        'recall_true': float(recall[0]),
        'f1_true': float(f1[0]),
        'precision_false': float(precision[1]),
        'recall_false': float(recall[1]),
        'f1_false': float(f1[1]),
        'precision_nei': float(precision[2]),
        'recall_nei': float(recall[2]),
        'f1_nei': float(f1[2]),
        'total_samples': len(data),
        'valid_samples': len(valid_idx),
        'inference_time_seconds': inference_time,
        'load_time_seconds': load_time,
        'avg_time_per_sample': inference_time / len(data),
    }
    
    with open('/home/ufb/FMD/eval_results_vllm.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n✓ Results saved to eval_results_vllm.json')

if __name__ == '__main__':
    main()
