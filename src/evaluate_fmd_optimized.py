#!/usr/bin/env python3
"""
FMD Evaluation Script - OPTIMIZED for Speed
- 4-bit quantization with Flash Attention v2
- Larger batch size (16)
- Shorter sequences for faster inference
"""

import torch
import argparse
import json
import os
import re
import random
import numpy as np
import time
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_prediction_finfact(text: str) -> str:
    """Parse FinFact prediction (True/False/NEI)."""
    text_lower = text.lower()

    # Look for "Prediction: X" pattern
    if re.search(r'prediction[:\s]*true', text_lower):
        return 'True'
    elif re.search(r'prediction[:\s]*false', text_lower):
        return 'False'
    elif re.search(r'prediction[:\s]*(nei|not enough)', text_lower):
        return 'NEI'

    # Fallback: check last part
    last_part = text_lower[-300:]
    if 'true' in last_part and 'false' not in last_part:
        return 'True'
    elif 'false' in last_part and 'true' not in last_part:
        return 'False'
    elif 'nei' in last_part:
        return 'NEI'

    return 'Unknown'


def clean_output(text: str) -> str:
    """Clean tokenizer artifacts."""
    text = text.replace('Ġ', ' ')
    text = text.replace('Ċ', '\n')
    text = re.sub(r' +', ' ', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='eval_results.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Load Data
    print(f"Loading data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Total samples: {len(data)}")

    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("Loading model with 4-bit quantization and Flash Attention v2...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=quant_config,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("✓ Flash Attention v2 enabled")
    except Exception as e:
        print(f"⚠ Flash Attention not available ({e}), using default attention")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=quant_config,
            device_map='auto',
            trust_remote_code=True,
        )

    model.eval()

    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

    # Prepare Prompts
    prompts = [f"Human:\n{item['instruction']}\n\nAssistant:\n" for item in data]
    gt_labels = []
    for item in data:
        if 'label' in item and item['label'] != 'Unknown':
            gt_labels.append(item['label'])
        else:
            gt_labels.append(parse_prediction_finfact(item.get('output', '')))

    known_label_idx = [i for i, l in enumerate(gt_labels) if l != 'Unknown']
    print(f"Samples with known labels: {len(known_label_idx)} / {len(data)}")

    # Batch Inference
    print(f"\n[OPTIMIZED SETTINGS]")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Max Seq Length: {args.max_seq_length}")
    print(f"\nRunning batch inference...")

    predictions = []
    raw_outputs = []
    
    start_time = time.time()
    pbar = tqdm(total=len(prompts), desc="Inference", unit="sample", dynamic_ncols=True)

    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
            ).to(device)

            # Generate with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode batch
            for j, output_ids in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[0]
                new_tokens = output_ids[input_len:]

                response = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                response = clean_output(response)

                pred = parse_prediction_finfact(response)
                predictions.append(pred)
                raw_outputs.append(response[:500])

            pbar.update(len(batch_prompts))

    pbar.close()
    total_time = time.time() - start_time

    # Calculate Metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    eval_idx = [i for i in range(len(predictions))
                if predictions[i] != 'Unknown' and gt_labels[i] != 'Unknown']
    valid_preds = [predictions[i] for i in eval_idx]
    valid_gts = [gt_labels[i] for i in eval_idx]

    print(f"\nInference Time: {total_time:.2f}s ({len(data)/total_time:.2f} samples/sec)")
    print(f"Total samples: {len(predictions)}")
    print(f"Evaluating on: {len(eval_idx)} samples")

    print(f"\nPrediction Distribution: {dict(Counter(predictions))}")
    print(f"Ground Truth Distribution: {dict(Counter(gt_labels))}")

    if valid_preds:
        accuracy = accuracy_score(valid_gts, valid_preds)
        labels = ['True', 'False', 'NEI']
        present_labels = [l for l in labels if l in valid_preds or l in valid_gts]
        macro_f1 = f1_score(valid_gts, valid_preds, labels=present_labels, average='macro', zero_division=0)        

        print(f"\n{'='*40}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1:  {macro_f1:.4f} ({macro_f1*100:.2f}%)")
        print(f"{'='*40}")

        print(f"\nClassification Report:")
        print(classification_report(valid_gts, valid_preds, labels=present_labels, zero_division=0))
    else:
        accuracy = 0
        macro_f1 = 0

    # Save Results
    print(f"\nSaving to {args.output_file}...")
    results = {
        'total_samples': len(data),
        'parsed_samples': len(valid_preds),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'total_time_sec': total_time,
        'samples_per_sec': len(data) / total_time,
        'predictions': [
            {'idx': i, 'pred': predictions[i], 'gt': gt_labels[i], 'output': raw_outputs[i]}
            for i in range(len(predictions))
        ]
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == '__main__':
    main()
