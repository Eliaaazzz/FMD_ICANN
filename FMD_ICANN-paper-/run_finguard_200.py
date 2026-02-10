#!/usr/bin/env python3
"""
FMDLlama Financial Misinformation Detection - FinGuard Evaluation (200 samples)
Replicating the FMDLlama paper results on FinGuard dataset
"""

import json
import torch
import re
import gc
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# Configuration
MODEL_PATH = "/home/ufb/models/FMDLlama3"
TEST_DATA_PATH = "/mnt/c/Users/Aufb/Desktop/FMD/processed_data/finguard/test.jsonl"
OUTPUT_PATH = "/mnt/c/Users/Aufb/Desktop/FMD/eval_finguard_1500_results.json"
NUM_SAMPLES = 1500

# Paper baseline (FMDLlama paper)
PAPER_ACCURACY = 0.7362
PAPER_MACRO_F1 = 0.6667


def parse_prediction(text):
    """Parse model output to get prediction (0=Fake, 1=True)"""
    if not text:
        return None

    # Clean up text - remove "assistant" prefix if present
    text = text.strip()
    if text.lower().startswith('assistant'):
        text = text[9:].strip()

    # Skip very short/truncated responses
    if len(text) < 3:
        return None

    text_lower = text.lower()

    # Look for direct answers first - most specific patterns
    # Pattern: "0. Fake" or "1. True" (expected format)
    if re.search(r'\b0\.?\s*fake\b', text_lower):
        return 0
    if re.search(r'\b1\.?\s*true\b', text_lower):
        return 1

    # Check for just the number at the very start (most reliable)
    first_char = text.strip()[:1]
    if first_char == '0':
        return 0
    if first_char == '1':
        return 1

    # Check for "fake" or "true" as standalone answer at start
    first_word = text_lower.split()[0] if text_lower.split() else ''
    if first_word in ['fake', 'fake.', 'fake,']:
        return 0
    if first_word in ['true', 'true.', 'true,']:
        return 1

    # Keywords for fake detection
    fake_keywords = ['misinformation', 'fabrication', 'fabricated', 'false', 'fake',
                     'misleading', 'inaccurate', 'not true', 'fiction', 'satirical',
                     'unreliable', 'hoax', 'debunked', 'conspiracy', 'unverified',
                     'disinformation', 'clickbait', 'sensationalized', 'baseless',
                     'unfounded', 'questionable', 'dubious', 'propaganda']

    # Keywords for true detection
    true_keywords = ['true', 'accurate', 'factual', 'real', 'correct', 'legitimate',
                     'verified', 'credible', 'reliable', 'authentic', 'valid', 'genuine',
                     'trustworthy', 'reputable', 'actual']

    # Reputable news sources - strong indicator of True
    reputable_sources = ['reuters', 'associated press', ' ap ', 'bloomberg', 'new york times',
                         'washington post', 'wall street journal', 'bbc', 'npr', 'cnn',
                         'financial times', 'economist', 'guardian', 'nyt', 'wsj']

    # Count keyword occurrences
    fake_count = sum(1 for kw in fake_keywords if kw in text_lower)
    true_count = sum(1 for kw in true_keywords if kw in text_lower)

    # Check for reputable source mentions (strong True signal)
    source_mentioned = any(src in text_lower for src in reputable_sources)
    if source_mentioned:
        true_count += 3

    # "reputable source" / "reliable source" / "reputable news agency" patterns
    if re.search(r'(reputable|reliable|credible|trustworthy)\s+(source|news|agency|outlet|publication)', text_lower):
        true_count += 3

    # "reports on real/actual events" pattern
    if re.search(r'reports?\s+on\s+(real|actual|true)\s+(event|fact|news)', text_lower):
        true_count += 2

    # "true story" or "real event"
    if re.search(r'(true|real|actual)\s+(story|event|news|article|report)', text_lower):
        true_count += 2

    # Adjust for negations - "not true" should count as fake
    if re.search(r'(not|isn\'t|is not|aren\'t|cannot be)\s+(true|accurate|factual|reliable|credible)', text_lower):
        fake_count += 3
        true_count -= 1

    # "this is true" or "appears true" or "is accurate"
    if re.search(r'(this|it|article|text|news|claim)\s+(is|appears|seems)\s+(to be\s+)?(true|accurate|factual|real)', text_lower):
        true_count += 2

    # "contains false information" pattern
    if re.search(r'contains?\s+(false|fake|misleading|inaccurate)\s+(information|claim|statement|content)', text_lower):
        fake_count += 2

    # If we see more indicators for one side
    if fake_count > true_count:
        return 0
    if true_count > fake_count:
        return 1

    # Look for the words (simple match)
    if 'fake' in text_lower and 'true' not in text_lower:
        return 0
    if 'true' in text_lower and 'fake' not in text_lower:
        return 1

    # If both present, check which comes first (usually the answer)
    fake_pos = text_lower.find('fake')
    true_pos = text_lower.find('true')

    if fake_pos != -1 and true_pos != -1:
        # Take the first mentioned one (usually the direct answer)
        if fake_pos < true_pos:
            return 0
        else:
            return 1

    # Last resort: check for general positive/negative sentiment about the article
    # "This article is from X" without negative words usually means it's considered true
    if re.search(r'(this|the)\s+(article|text|news)\s+(is\s+)?(from|by)\s+', text_lower):
        if fake_count == 0:  # No fake indicators found
            return 1

    return None


def load_model():
    """Load FMDLlama3 with 4-bit quantization"""
    print(f"Loading model from {MODEL_PATH} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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

    return model, tokenizer


def run_inference(model, tokenizer, prompt, max_new_tokens=50):
    """Run single inference"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


def main():
    print("=" * 70)
    print("FMDLlama Financial Misinformation Detection")
    print("FinGuard Dataset Evaluation - 200 Samples")
    print("=" * 70)

    # Load test data
    print(f"\n[1/4] Loading test data from {TEST_DATA_PATH}...")
    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    # Take only NUM_SAMPLES
    test_data = test_data[:NUM_SAMPLES]
    print(f"Using {len(test_data)} samples for evaluation")

    # Count label distribution
    label_dist = {}
    for item in test_data:
        label = item.get('label', 'unknown')
        label_dist[label] = label_dist.get(label, 0) + 1
    print(f"Label distribution: {label_dist}")

    # Load model
    print("\n[2/4] Loading FMDLlama3 model...")
    start_load = time.time()
    model, tokenizer = load_model()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    # Run inference
    print("\n[3/4] Running inference...")
    predictions = []
    ground_truths = []
    sample_outputs = []
    failed = 0

    start_inference = time.time()

    for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
        # Get the formatted text (already has Llama 3.1 format)
        # Add assistant header to signal the model to start responding
        prompt = item.get('text', '')
        if not prompt.endswith('<|eot_id|>'):
            prompt = prompt.rstrip() + '<|eot_id|>'
        # Add explicit format instruction and start with the answer format
        prompt = prompt + '<|start_header_id|>assistant<|end_header_id|>\n\n'

        # Get ground truth label
        label_str = item.get('label', '')
        if label_str in ['Real', 'True']:
            true_label = 1
        elif label_str == 'Fake':
            true_label = 0
        else:
            continue

        try:
            # Use shorter max_new_tokens to encourage concise answers
            response = run_inference(model, tokenizer, prompt, max_new_tokens=20)
            pred_label = parse_prediction(response)

            # Save all outputs for debugging
            sample_outputs.append({
                'index': i,
                'true_label': label_str,
                'pred_label': 'Real' if pred_label == 1 else ('Fake' if pred_label == 0 else 'Unknown'),
                'response': response[:200],
                'correct': (pred_label == 1 and label_str in ['True', 'Real']) or (pred_label == 0 and label_str == 'Fake')
            })

            if pred_label is None:
                failed += 1
                pred_label = -1  # Unknown

            predictions.append(pred_label)
            ground_truths.append(true_label)

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            failed += 1

        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    inference_time = time.time() - start_inference

    # Calculate metrics
    print("\n[4/4] Computing metrics...")

    # Filter out failed predictions for accuracy calculation
    valid_idx = [i for i in range(len(predictions)) if predictions[i] != -1]
    valid_preds = [predictions[i] for i in valid_idx]
    valid_gts = [ground_truths[i] for i in valid_idx]

    if not valid_preds:
        print("ERROR: No valid predictions!")
        return

    accuracy = accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average='macro', zero_division=0)

    # Classification report
    target_names = ['Fake (0)', 'Real (1)']
    report = classification_report(valid_gts, valid_preds, target_names=target_names, zero_division=0)
    conf_matrix = confusion_matrix(valid_gts, valid_preds)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nDataset: FinGuard (Financial Truth Guard)")
    print(f"Total samples: {NUM_SAMPLES}")
    print(f"Valid predictions: {len(valid_preds)}")
    print(f"Failed to parse: {failed}")

    print(f"\n{'Metric':<20} {'Our Result':<15} {'Paper Result':<15} {'Delta':<15}")
    print("-" * 65)
    print(f"{'Accuracy':<20} {accuracy:.4f}         {PAPER_ACCURACY:.4f}         {accuracy - PAPER_ACCURACY:+.4f}")
    print(f"{'Macro F1':<20} {macro_f1:.4f}         {PAPER_MACRO_F1:.4f}         {macro_f1 - PAPER_MACRO_F1:+.4f}")

    print(f"\nClassification Report:\n{report}")

    print("Confusion Matrix (rows=true, cols=pred):")
    print("         Fake  Real")
    for i, row in enumerate(conf_matrix):
        label = 'Fake' if i == 0 else 'Real'
        print(f"{label:>8} {row[0]:5d} {row[1]:5d}")

    # Sample outputs - show first 20 for debugging
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS (first 20)")
    print("=" * 70)
    for sample in sample_outputs[:20]:
        correct = "OK" if sample.get('correct', False) else "WRONG"
        print(f"\n[{sample['index']}] True: {sample['true_label']} | Pred: {sample['pred_label']} [{correct}]")
        print(f"     Response: {sample['response'][:100]}...")

    # Show parsing failure stats
    unknown_samples = [s for s in sample_outputs if s['pred_label'] == 'Unknown']
    if unknown_samples:
        print(f"\n\nFailed to parse {len(unknown_samples)} samples. Examples:")
        for s in unknown_samples[:5]:
            print(f"  [{s['index']}] Response: {s['response'][:80]}...")

    # Timing info
    print("\n" + "=" * 70)
    print("TIMING INFO")
    print("=" * 70)
    print(f"Model load time: {load_time:.1f}s")
    print(f"Inference time: {inference_time:.1f}s")
    print(f"Avg time per sample: {inference_time/NUM_SAMPLES:.2f}s")
    print(f"Samples/second: {NUM_SAMPLES/inference_time:.2f}")

    # Save results
    results = {
        'dataset': 'FinGuard',
        'num_samples': NUM_SAMPLES,
        'valid_samples': len(valid_preds),
        'failed_samples': failed,
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'paper_accuracy': PAPER_ACCURACY,
        'paper_macro_f1': PAPER_MACRO_F1,
        'confusion_matrix': conf_matrix.tolist(),
        'load_time_seconds': load_time,
        'inference_time_seconds': inference_time,
        'avg_time_per_sample': inference_time / NUM_SAMPLES,
        'sample_outputs': sample_outputs
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
