#!/usr/bin/env python3
"""
Script to download and split the full Fin-Fact dataset according to the paper.
Paper: FMDLlama (Train: 1562, Val: 391, Test: 1304)
"""
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

# Load Fin-Fact dataset from HuggingFace
print("Loading Fin-Fact dataset from HuggingFace...")
ds = load_dataset('amanrangapur/Fin-Fact')
df = pd.DataFrame(ds['train'])
print(f"Total samples: {len(df)}")

# Print column names
print(f"Columns: {list(df.columns)}")

# Check for evidence/justification column and remove empty ones
# Based on the paper: "remove the data without evidence"
if 'justification' in df.columns:
    df_filtered = df[df['justification'].notna() & (df['justification'] != '')]
elif 'evidence' in df.columns:
    df_filtered = df[df['evidence'].notna() & (df['evidence'] != '')]
else:
    print("No evidence column found, checking available columns...")
    print(df.head())
    df_filtered = df

print(f"After filtering (removing samples without evidence): {len(df_filtered)}")

# Split according to paper ratios: 1562 train, 391 val, 1304 test
# Total: 3257 samples (1562+391+1304)
# Ratio: train=48%, val=12%, test=40%
total = len(df_filtered)
print(f"Splitting {total} samples...")

# First split: separate test (40%)
train_val, test = train_test_split(df_filtered, test_size=0.4, random_state=42)
# Second split: separate train/val (48% train, 12% val of total = 80/20 of remaining)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Paper expects: Train: 1562, Val: 391, Test: 1304")

# Create output directory
os.makedirs('./data/full_data', exist_ok=True)

# Define instruction template (from paper)
pre_instruct = '''Please determine whether the claim is True, False, or Not Enough Information (NEI) based on contextual information, and provide an appropriate explanation.
The answer need to use the following format:
Prediction: [True, or False, or NEI]
Explanation: [Explain why the above prediction was made]
'''

def create_instruction_data(df_split, output_path, split_name):
    print(f"Creating {split_name} instruction data...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df_split.iterrows():
            claim = row.get('claim', '')
            justification = row.get('justification', row.get('explanation', ''))
            label = str(row.get('label', ''))
            evidence = row.get('evidence', row.get('explanation', ''))
            
            instruction = f"Task: {pre_instruct} Claim: {claim}\ncontextual information: {justification}"
            output = f"Prediction: {label}.\nExplanation: {evidence}"
            
            data = {
                'instruction': instruction,
                'input': '',
                'output': output,
                'label': label
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"  Written to {output_path}")

# Create instruction data files
create_instruction_data(train, './data/full_data/FMD_train_full.json', 'train')
create_instruction_data(val, './data/full_data/FMD_val_full.json', 'val')
create_instruction_data(test, './data/full_data/FMD_test_full.json', 'test')

print("\nDone! Created files:")
print("  - data/full_data/FMD_train_full.json")
print("  - data/full_data/FMD_val_full.json") 
print("  - data/full_data/FMD_test_full.json")
