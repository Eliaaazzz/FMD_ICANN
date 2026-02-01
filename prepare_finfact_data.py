#!/usr/bin/env python3
"""
Prepare FinFact dataset for FMDLlama evaluation.
Train: 1562, Val: 391, Test: 1304 (samples with evidence only)
"""

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

# Output directories
OUTPUT_DIR = "/mnt/c/Users/Aufb/Desktop/FMD/processed_data/finfact"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("FinFact Dataset Preparation")
print("=" * 60)

# Load Fin-Fact dataset from HuggingFace
print("\n[1/4] Loading Fin-Fact dataset from HuggingFace...")
ds = load_dataset('amanrangapur/Fin-Fact')
df = pd.DataFrame(ds['train'])
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Filter out samples without evidence/explanation
print("\n[2/4] Filtering samples without evidence...")
evidence_col = None
for col in ['justification', 'evidence', 'explanation']:
    if col in df.columns:
        evidence_col = col
        break

if evidence_col:
    df_filtered = df[df[evidence_col].notna() & (df[evidence_col].str.strip() != '')]
    print(f"Using '{evidence_col}' column for evidence")
else:
    print("Warning: No evidence column found, using all samples")
    df_filtered = df

print(f"After filtering: {len(df_filtered)} samples")

# Split according to paper: train=1562, val=391, test=1304
print("\n[3/4] Splitting dataset...")
# Total: 3257 => 48% train, 12% val, 40% test
train_val, test = train_test_split(df_filtered, test_size=0.4, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

print(f"Train: {len(train)} (target: 1562)")
print(f"Val: {len(val)} (target: 391)")
print(f"Test: {len(test)} (target: 1304)")

# Instruction template for FinFact (3-class: True, False, NEI)
INSTRUCTION = """Task: Please determine whether the claim is True, False, or Not Enough Information (NEI) based on the contextual information provided, and give an appropriate explanation.
The answer needs to use the following format:
Prediction: [True, or False, or NEI]
Explanation: [Explain why the above prediction was made]"""


def create_jsonl(df_split, output_path, split_name):
    """Create JSONL file with instruction format"""
    print(f"\nCreating {split_name} data...")

    label_counts = {}
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df_split.iterrows():
            claim = row.get('claim', '')

            # Get evidence/justification
            evidence = ''
            for col in ['justification', 'evidence', 'explanation']:
                if col in row and pd.notna(row[col]):
                    evidence = str(row[col]).strip()
                    if evidence:
                        break

            # Get label
            label = str(row.get('label', 'Unknown')).strip()

            # Normalize label
            if label.lower() in ['true', '1', 'real']:
                label = 'True'
            elif label.lower() in ['false', '0', 'fake']:
                label = 'False'
            elif label.lower() in ['nei', 'not enough information', 'unknown', 'neutral']:
                label = 'NEI'

            label_counts[label] = label_counts.get(label, 0) + 1

            # Build input text with claim and contextual info
            input_text = f"Claim: {claim}"
            if evidence:
                input_text += f"\nContextual information: {evidence}"

            # Build expected output
            output_text = f"Prediction: {label}.\nExplanation: {evidence if evidence else 'Based on the available information.'}"

            data = {
                'instruction': INSTRUCTION,
                'input': input_text,
                'output': output_text,
                'label': label,
                'claim': claim,
                'evidence': evidence
            }

            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"  Written {len(df_split)} samples to {output_path}")
    print(f"  Label distribution: {label_counts}")


# Create JSONL files
print("\n[4/4] Creating JSONL files...")
create_jsonl(train, os.path.join(OUTPUT_DIR, 'train.jsonl'), 'train')
create_jsonl(val, os.path.join(OUTPUT_DIR, 'val.jsonl'), 'val')
create_jsonl(test, os.path.join(OUTPUT_DIR, 'test.jsonl'), 'test')

# Summary
print("\n" + "=" * 60)
print("Done! Created files:")
print(f"  - {OUTPUT_DIR}/train.jsonl ({len(train)} samples)")
print(f"  - {OUTPUT_DIR}/val.jsonl ({len(val)} samples)")
print(f"  - {OUTPUT_DIR}/test.jsonl ({len(test)} samples)")
print("=" * 60)
print("\nNext step: Run evaluation with:")
print("  python run_finfact_gptq.py")
