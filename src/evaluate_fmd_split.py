#!/usr/bin/env python3
"""
FMD Evaluation Script - Split by Dataset (FinFact vs FinGuard)
Optimized for speed:
- 4-bit quantization
- Short generation for FinGuard (classification only)
- Longer generation for FinFact (explanation needed)
"""

import torch
import argparse
import json
import os
import sys
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

DEFAULT_MODEL_PATH = '/home/ufb/models/FMDLlama3'
DEFAULT_DATA_PATH = '/home/ufb/FMD/processed_data/finguard/test.jsonl'

def parse_prediction(text: str) -> str:
    """Parse prediction from model output."""
    text_lower = text.lower()
    
    if 'prediction: 1. true' in text_lower or 'prediction: true' in text_lower:
        return 'true'
    if 'prediction: 0. false' in text_lower or 'prediction: false' in text_lower:
        return 'false'
    if 'prediction: 2. nei' in text_lower or 'prediction: nei' in text_lower or 'prediction: neutral' in text_lower:
        return 'nei'
        
    if 'true' in text_lower and 'false' not in text_lower:
        return 'true'
    elif 'false' in text_lower and 'true' not in text_lower:
        return 'false'
    elif 'nei' in text_lower or 'not enough information' in text_lower or 'neutral' in text_lower:
        return 'nei'
    elif 'fake' in text_lower: # FinGuard specific
        return 'false'
        
    return 'unknown'

def normalize_label(label):
    """Normalize ground truth labels."""
    if not label: return 'unknown'
    l = str(label).lower().strip()
    if 'true' in l: return 'true'
    if 'false' in l or 'fake' in l: return 'false'
    if 'nei' in l or 'neutral' in l: return 'nei'
    return 'unknown'

def format_prompt_llama3(item: dict) -> str:
    """Format prompt using Llama 3 template."""
    instruction = item.get('instruction', '')
    input_text = item.get('input', item.get('input_text', item.get('context', '')))
    
    if not input_text:
        if 'claim' in item:
            input_text = f"Claim: {item['claim']}\nClaim summaries: {item.get('summaries', '')}\nContextual information: {item.get('contextual', '')}"
        elif 'text' in item and 'instruction' in item: 
            input_text = item['text']
            
    system_prompt = "You are a financial misinformation detection expert."
    
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>".format(system_prompt=system_prompt)
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}".format(instruction=instruction)
    
    if input_text:
        prompt += f"\n\n{input_text}".format(input_text=input_text)
        
    prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

DEFAULT_DATA_PATH = '/home/ufb/FMD/processed_data/finguard/test.jsonl'

def identify_dataset(item: dict) -> str:
    """Identify if item is FinFact or FinGuard."""
    # Robust check based on data fields
    if 'claim' in item:
        return 'finfact'
    elif 'text' in item:
        return 'finguard'
    
    # Fallback to instruction analysis
    instruction = item.get('instruction', '').lower()
    if 'explanation' in instruction or 'claim' in instruction:
        return 'finfact'
    else:
        return 'finguard'

def evaluate_subset(model, tokenizer, data, dataset_name, batch_size=8, max_new_tokens=64):
    if not data:
        print(f"Skipping {dataset_name} (no data)")
        return 0.0, 0.0, []

    print(f"\nEvaluating {dataset_name.upper()} ({len(data)} samples)...")
    prompts = [format_prompt_llama3(item) for item in data]
    gts = []
    
    for item in data:
        if 'label' in item:
            gts.append(normalize_label(item['label']))
        elif 'output' in item:
            gts.append(normalize_label(parse_prediction(item['output'])))
        else:
            gts.append('unknown')

    predictions = []
    temp_file = f"eval_{dataset_name}.tmp"
    
    # Resume
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            predictions = json.load(f)
        print(f"Resuming {dataset_name} from {len(predictions)} samples...")

    with torch.no_grad():
        for i in tqdm(range(len(predictions), len(prompts), batch_size), desc=dataset_name):
            batch_prompts = prompts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048,
            ).to('cuda')
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            for j, output_ids in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[0]
                generated_text = tokenizer.decode(
                    output_ids[input_len:], 
                    skip_special_tokens=True
                )
                pred = parse_prediction(generated_text)
                predictions.append(pred)
                
                # Debug print for first few samples
                if len(predictions) <= 5:
                    print(f"\n[DEBUG Sample {len(predictions)}]")
                    print(f"Generated: {generated_text!r}")
                    print(f"Parsed: {pred}")
                    print(f"Ground Truth: {gts[len(predictions)-1]}")
            
            # Save checkpoint
            with open(temp_file, 'w') as f:
                json.dump(predictions, f)

    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

    valid_idx = [i for i in range(len(predictions)) 
                 if predictions[i] != 'unknown' and gts[i] != 'unknown']
    
    if not valid_idx:
        return 0.0, 0.0, []

    valid_preds = [predictions[i] for i in valid_idx]
    valid_gts = [gts[i] for i in valid_idx]

    acc = accuracy_score(valid_gts, valid_preds)
    f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)
    
    print(f"\n--- {dataset_name.upper()} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    return acc, f1, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--samples_per_task', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    try:
        with open(args.data_path, 'r') as f:
            full_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Data file not found.")
        return

    # Split data
    finfact_data = []
    finguard_data = []
    
    for item in full_data:
        if identify_dataset(item) == 'finfact':
            finfact_data.append(item)
        else:
            finguard_data.append(item)
            
    print(f"Found {len(finfact_data)} FinFact samples and {len(finguard_data)} FinGuard samples.")
    
    # Subset
    finfact_subset = finfact_data[:args.samples_per_task]
    finguard_subset = finguard_data[:args.samples_per_task]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Loading model (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    # Run Evaluations
    # 1. FinGuard (Fast - only classification)
    # Using small max_new_tokens because we only need "True/Fake"
    fg_acc, fg_f1, _ = evaluate_subset(
        model, tokenizer, finguard_subset, 'finguard', 
        batch_size=args.batch_size, max_new_tokens=10
    )

    # 2. FinFact (Slower - needs explanation)
    ff_acc, ff_f1, _ = evaluate_subset(
        model, tokenizer, finfact_subset, 'finfact', 
        batch_size=args.batch_size, max_new_tokens=64
    )
    
    print("\n" + "="*50)
    print("FINAL REPORT (100 samples each)")
    print("="*50)
    print(f"FinGuard Accuracy: {fg_acc:.4f}")
    print(f"FinFact Accuracy:  {ff_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
