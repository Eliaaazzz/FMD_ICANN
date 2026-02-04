#!/usr/bin/env python3
"""
FMD Evaluation Script - Fixed for Llama 3 Format & Input Handling
"""

import torch
import argparse
import json
import os
import re
import sys
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Default Paths (can be overridden by args)
DEFAULT_MODEL_PATH = '/home/ufb/models/FMDLlama3'
DEFAULT_DATA_PATH = '/home/ufb/FMD/data/full_data/FMD_test_full.json'

def parse_prediction(text: str) -> str:
    """Parse prediction from model output."""
    text_lower = text.lower()
    
    # Look for explicit "Prediction:" pattern first
    if 'prediction: 1. true' in text_lower or 'prediction: true' in text_lower:
        return 'true'
    if 'prediction: 0. false' in text_lower or 'prediction: false' in text_lower:
        return 'false'
    if 'prediction: 2. nei' in text_lower or 'prediction: nei' in text_lower or 'prediction: neutral' in text_lower:
        return 'nei'
        
    # Fallback: simple keyword search in the generated text
    if 'true' in text_lower and 'false' not in text_lower:
        return 'true'
    elif 'false' in text_lower and 'true' not in text_lower:
        return 'false'
    elif 'nei' in text_lower or 'not enough information' in text_lower or 'neutral' in text_lower:
        return 'nei'
        
    return 'unknown'

def normalize_label(label):
    """Normalize ground truth labels."""
    if not label: return 'unknown'
    l = str(label).lower().strip()
    if 'true' in l: return 'true'
    if 'false' in l: return 'false'
    if 'nei' in l or 'neutral' in l: return 'nei'
    return 'unknown'

def format_prompt_llama3(item: dict) -> str:
    """
    Format prompt using Llama 3 template.
    Ref: prepare_fmd_data.py
    """
    
    # 1. Extract Instruction and Input
    instruction = item.get('instruction', '')
    input_text = item.get('input', item.get('input_text', item.get('context', '')))
    
    # If input is empty, maybe instruction contains everything?
    if not input_text:
        if 'claim' in item:
            input_text = f"Claim: {item['claim']}\nClaim summaries: {item.get('summaries', '')}\nContextual information: {item.get('contextual', '')}"
        elif 'text' in item and 'instruction' in item: 
            input_text = item['text']
            
    # 2. Construct Prompt with Special Tokens
    system_prompt = "You are a financial misinformation detection expert."
    
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
    
    if input_text:
        prompt += f"\n\n{input_text}"
        
    prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt

def main():
    # ... (omitted args parsing) ...

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--output_file', type=str, default='eval_results_fixed.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--subset', type=int, default=None, help='Number of samples to evaluate')
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    try:
        with open(args.data_path, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        return

    if args.subset:
        print(f"Using subset of {args.subset} samples")
        data = data[:args.subset]

    print(f"Total samples to process: {len(data)}")
    
    # Debug: Print first prompt construction
    print("\n--- Sample Prompt Preview ---")
    print(format_prompt_llama3(data[0]))
    print("-----------------------------\\n")

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

    # Prepare prompts
    prompts = [format_prompt_llama3(item) for item in data]
    
    # Prepare Ground Truth
    gts = []
    for item in data:
        if 'label' in item:
            gts.append(normalize_label(item['label']))
        elif 'output' in item:
            gts.append(normalize_label(parse_prediction(item['output'])))
        else:
            gts.append('unknown')

    predictions = []
    
    # Check for existing partial results
    temp_file = args.output_file + ".tmp"
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            predictions = json.load(f)
        print(f"Resuming from {len(predictions)} saved predictions...")

    print(f"Running inference (Batch Size: {args.batch_size})...")
    
    with torch.no_grad():
        for i in tqdm(range(len(predictions), len(prompts), args.batch_size)):
            batch_prompts = prompts[i:i + args.batch_size]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048,
            ).to('cuda')
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
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
            
            # Save progress after each batch
            with open(temp_file, 'w') as f:
                json.dump(predictions, f)
            
            print(f"Processed {len(predictions)}/{len(prompts)} samples...")

    # Cleanup temp file on completion
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Evaluation
    valid_idx = [i for i in range(len(predictions)) 
                 if predictions[i] != 'unknown' and gts[i] != 'unknown']
    
    if not valid_idx:
        print("No valid predictions found!")
        return

    valid_preds = [predictions[i] for i in valid_idx]
    valid_gts = [gts[i] for i in valid_idx]

    acc = accuracy_score(valid_gts, valid_preds)
    f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)

    print("\n" + "="*50)
    print("RESULTS (Llama 3 Format)")
    print("="*50)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1: {f1:.4f} ({f1*100:.2f}%)")
    print("-" * 50)
    print(classification_report(valid_gts, valid_preds, zero_division=0))
    print("="*50)

    # Save
    results = {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'predictions': [
            {'idx': i, 'pred': predictions[i], 'gt': gts[i]} 
            for i in range(len(predictions))
        ]
    }
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
