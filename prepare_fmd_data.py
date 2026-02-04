#!/usr/bin/env python3
"""
FMD (Financial Misinformation Detection) Data Preparation Script
=================================================================

This script prepares training data for replicating the FMDLlama model as described
in the research paper. It processes two datasets:
1. FinFact - Financial fact-checking with explanations
2. FinGuard - Financial fake/real news classification

Author: Data Engineering Pipeline for FMDLlama Replication
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# GitHub Repository URLs
FINFACT_REPO_URL = "https://github.com/IIT-DM/Fin-Fact"
FINGUARD_REPO_URL = "https://github.com/carlos-gmartin/Financial-Truth-Guard"

# Local paths (adjust these to your directory structure)
BASE_DATA_DIR = Path("./raw_data")
FINFACT_DIR = BASE_DATA_DIR / "Fin-Fact"
FINGUARD_DIR = BASE_DATA_DIR / "Financial-Truth-Guard"

# Output directory
OUTPUT_DIR = Path("./processed_data")

# Random seed for reproducibility (paper methodology)
RANDOM_SEED = 42

# FinGuard sampling size (as per paper: 2,500 from each class)
FINGUARD_SAMPLE_SIZE = 2500

# Train/Val/Test split ratios (Paper: 2900/600/1500 = 58%/12%/30%)
TRAIN_RATIO = 0.58
VAL_RATIO = 0.12
TEST_RATIO = 0.30

# ==============================================================================
# INSTRUCTION TEMPLATES (From Paper Table 2)
# ==============================================================================

# Task 1: FinFact Template (Requires Explanation)
FINFACT_INSTRUCTION_TEMPLATE = """Task: Please determine whether the claim is 0. False, 1. True, or 2. Not Enough Information (NEI) based on contextual information, and provide an appropriate explanation.
The answer needs to use the following format:
Prediction: [0. False, 1. True, or 2. NEI]
Explanation: [Explain why the above prediction was made]"""

FINFACT_INPUT_TEMPLATE = """Claim: {claim}
Claim summaries: {summaries}
Contextual information: {contextual}"""

# Task 2: FinGuard Template (Direct Classification)
FINGUARD_INSTRUCTION_TEMPLATE = """Task: Please determine whether the text is 0. Fake or 1. True. Answer directly without explanations."""

FINGUARD_INPUT_TEMPLATE = """Text: {text}"""

# ==============================================================================
# LLAMA 3.1 FINE-TUNING FORMAT
# ==============================================================================

# Option 1: Simple text format
LLAMA_TEXT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial misinformation detection expert.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

# Option 2: Messages format (for chat fine-tuning)
def create_messages_format(instruction: str, input_text: str, output: str) -> Dict:
    """Create the messages format compatible with Llama 3.1 fine-tuning."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a financial misinformation detection expert."
            },
            {
                "role": "user", 
                "content": f"{instruction}\n\n{input_text}"
            },
            {
                "role": "assistant",
                "content": output
            }
        ]
    }


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class FinFactSample:
    """Represents a single FinFact data sample."""
    claim: str
    summaries: str
    contextual: str
    label: str  # True, False, NEI
    explanation: str
    
    def has_evidence(self) -> bool:
        """Check if sample has valid evidence/explanation (paper requirement)."""
        return bool(self.explanation and self.explanation.strip())


@dataclass  
class FinGuardSample:
    """Represents a single FinGuard data sample."""
    text: str
    label: str  # Fake or True
    source: str  # 'real' or 'fake' from original dataset


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_finfact_data(data_path: Path) -> List[FinFactSample]:
    """
    Load FinFact dataset.
    
    Expected FinFact repository structure:
    Fin-Fact/
    ├── data/
    │   ├── train.json (or train.jsonl)
    │   ├── dev.json (or val.json)
    │   └── test.json
    └── README.md
    
    The FinFact dataset typically contains fields like:
    - claim: The financial claim to verify
    - evidence: Supporting evidence/explanation
    - label: Verdict (true/false/NEI)
    - claim_summary or summaries: Summary of the claim
    - context or contextual_info: Contextual information
    """
    samples = []
    
    # Try different possible file locations
    possible_paths = [
        data_path / "data" / "train.json",
        data_path / "data" / "finfact_train.json",
        data_path / "train.json",
        data_path / "data" / "train.jsonl",
        data_path / "finfact.json",
    ]
    
    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = p
            print(f"  Found FinFact data at: {p}")
            break
    
    if data_file is None:
        print(f"  WARNING: FinFact data not found. Tried paths: {possible_paths}")
        print(f"  Please clone the repository: git clone {FINFACT_REPO_URL}")
        return samples
    
    # Load the data
    with open(data_file, 'r', encoding='utf-8') as f:
        if data_file.suffix == '.jsonl':
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get('data', data.get('samples', [data]))
    
    # Parse each sample
    for item in data:
        # Handle different field naming conventions
        claim = item.get('claim', item.get('text', ''))
        summaries = item.get('claim_summary', item.get('summaries', item.get('summary', '')))
        contextual = item.get('context', item.get('contextual_info', item.get('evidence_text', '')))
        label = item.get('label', item.get('verdict', item.get('class', '')))
        explanation = item.get('evidence', item.get('explanation', item.get('rationale', '')))
        
        # Normalize label
        label_lower = str(label).lower()
        if 'true' in label_lower or label_lower == '1':
            normalized_label = 'True'
        elif 'false' in label_lower or label_lower == '0':
            normalized_label = 'False'
        else:
            normalized_label = 'NEI'
        
        sample = FinFactSample(
            claim=str(claim),
            summaries=str(summaries) if summaries else '',
            contextual=str(contextual) if contextual else '',
            label=normalized_label,
            explanation=str(explanation) if explanation else ''
        )
        samples.append(sample)
    
    return samples


def load_finguard_data(data_path: Path) -> Tuple[List[FinGuardSample], List[FinGuardSample]]:
    """
    Load FinGuard (Financial Truth Guard) dataset.
    Reads from CSV files in Pilot/data/finance_dataset/
    """
    import csv
    
    real_samples = []
    fake_samples = []
    
    # Correct paths based on repository structure
    # Use glob to find the files recursively if needed
    real_paths = [
        data_path / "Pilot/data/finance_dataset/Finance_TRUE.csv",
        data_path / "data/finance_dataset/Finance_TRUE.csv", 
        data_path / "Finance_TRUE.csv"
    ]
    
    fake_paths = [
        data_path / "Pilot/data/finance_dataset/Finance_FAKE.csv",
        data_path / "data/finance_dataset/Finance_FAKE.csv",
        data_path / "Finance_FAKE.csv"
    ]
    
    def load_csv(filepath: Path, label: str) -> List[FinGuardSample]:
        samples = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSV typically has 'text' column
                    text = row.get('text', row.get('title', '')) 
                    # If 'text' is missing, try combining title + text
                    if not text and 'title' in row:
                        text = row['title']
                        
                    if text:
                        samples.append(FinGuardSample(
                            text=str(text),
                            label=label,
                            source=label.lower()
                        ))
            print(f"  Loaded {len(samples)} {label} samples from {filepath}")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
        return samples

    # Find and load Real
    for p in real_paths:
        if p.exists():
            real_samples = load_csv(p, 'True')
            break
            
    # Find and load Fake
    for p in fake_paths:
        if p.exists():
            fake_samples = load_csv(p, 'Fake')
            break
    
    if not real_samples and not fake_samples:
        print(f"  WARNING: FinGuard data not found in {data_path}")
        print(f"  Expected paths like: {real_paths[0]}")
    
    return real_samples, fake_samples


# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================

def filter_finfact_samples(samples: List[FinFactSample]) -> List[FinFactSample]:
    """
    Filter FinFact samples according to paper methodology:
    "We remove the data without evidence (i.e., explanation)."
    """
    original_count = len(samples)
    filtered = [s for s in samples if s.has_evidence()]
    removed_count = original_count - len(filtered)
    
    print(f"  FinFact filtering: {original_count} -> {len(filtered)} samples")
    print(f"  Removed {removed_count} samples without evidence/explanation")
    
    return filtered


def sample_finguard_data(
    real_samples: List[FinGuardSample],
    fake_samples: List[FinGuardSample],
    sample_size: int = FINGUARD_SAMPLE_SIZE
) -> List[FinGuardSample]:
    """
    Sample FinGuard data according to paper methodology:
    "We separately extracted 2,500 data points from both real and fake data."
    """
    random.seed(RANDOM_SEED)
    
    # Sample from each class
    if len(real_samples) >= sample_size:
        sampled_real = random.sample(real_samples, sample_size)
    else:
        print(f"  WARNING: Only {len(real_samples)} real samples available (need {sample_size})")
        sampled_real = real_samples
    
    if len(fake_samples) >= sample_size:
        sampled_fake = random.sample(fake_samples, sample_size)
    else:
        print(f"  WARNING: Only {len(fake_samples)} fake samples available (need {sample_size})")
        sampled_fake = fake_samples
    
    combined = sampled_real + sampled_fake
    random.shuffle(combined)
    
    print(f"  FinGuard sampling: {len(sampled_real)} real + {len(sampled_fake)} fake = {len(combined)} total")
    
    return combined


def split_data(
    samples: List,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Tuple[List, List, List]:
    """Split data into train/val/test sets."""
    random.seed(RANDOM_SEED)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_set = shuffled[:train_end]
    val_set = shuffled[train_end:val_end]
    test_set = shuffled[val_end:]
    
    print(f"  Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_set, val_set, test_set


# ==============================================================================
# FORMAT CONVERSION FUNCTIONS
# ==============================================================================

def format_finfact_sample(sample: FinFactSample, output_format: str = 'text') -> Dict:
    """Convert FinFact sample to Llama 3.1 fine-tuning format."""
    
    # Create the input using the template
    input_text = FINFACT_INPUT_TEMPLATE.format(
        claim=sample.claim,
        summaries=sample.summaries,
        contextual=sample.contextual
    )
    
    # Create the output (prediction + explanation)
    label_map = {'True': '1. True', 'False': '0. False', 'NEI': '2. NEI'}
    formatted_label = label_map.get(sample.label, sample.label)
    
    output_text = f"Prediction: {formatted_label}\nExplanation: {sample.explanation}"
    
    if output_format == 'messages':
        return create_messages_format(
            instruction=FINFACT_INSTRUCTION_TEMPLATE,
            input_text=input_text,
            output=output_text
        )
    else:  # text format
        return {
            "text": LLAMA_TEXT_TEMPLATE.format(
                instruction=FINFACT_INSTRUCTION_TEMPLATE,
                input=input_text,
                output=output_text
            ),
            "task": "finfact",
            "label": sample.label
        }


def format_finguard_sample(sample: FinGuardSample, output_format: str = 'text') -> Dict:
    """Convert FinGuard sample to Llama 3.1 fine-tuning format."""
    
    # Create the input using the template
    input_text = FINGUARD_INPUT_TEMPLATE.format(text=sample.text)
    
    # Create the output (direct answer)
    label_map = {'True': '1. True', 'Fake': '0. Fake'}
    output_text = label_map.get(sample.label, sample.label)
    
    if output_format == 'messages':
        return create_messages_format(
            instruction=FINGUARD_INSTRUCTION_TEMPLATE,
            input_text=input_text,
            output=output_text
        )
    else:  # text format
        return {
            "text": LLAMA_TEXT_TEMPLATE.format(
                instruction=FINGUARD_INSTRUCTION_TEMPLATE,
                input=input_text,
                output=output_text
            ),
            "task": "finguard",
            "label": sample.label
        }


# ==============================================================================
# OUTPUT FUNCTIONS
# ==============================================================================

def save_jsonl(data: List[Dict], filepath: Path):
    """Save data to JSONL format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} samples to {filepath}")


def generate_dataset_stats(
    finfact_train: List, finfact_val: List, finfact_test: List,
    finguard_train: List, finguard_val: List, finguard_test: List
) -> Dict:
    """Generate statistics about the processed datasets."""
    stats = {
        "finfact": {
            "train": len(finfact_train),
            "val": len(finfact_val),
            "test": len(finfact_test),
            "total": len(finfact_train) + len(finfact_val) + len(finfact_test)
        },
        "finguard": {
            "train": len(finguard_train),
            "val": len(finguard_val),
            "test": len(finguard_test),
            "total": len(finguard_train) + len(finguard_val) + len(finguard_test)
        },
        "combined": {
            "train": len(finfact_train) + len(finguard_train),
            "val": len(finfact_val) + len(finguard_val),
            "test": len(finfact_test) + len(finguard_test),
            "total": (len(finfact_train) + len(finfact_val) + len(finfact_test) +
                     len(finguard_train) + len(finguard_val) + len(finguard_test))
        },
        "generated_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED
    }
    return stats


# ==============================================================================
# VERIFICATION / DUMMY CHECK FUNCTIONS
# ==============================================================================

def dummy_check_finfact():
    """Print a sample FinFact prompt for manual verification against Paper Table 2."""
    print("\n" + "="*80)
    print("DUMMY CHECK: FinFact Sample (Task 1)")
    print("="*80)
    
    sample = FinFactSample(
        claim="Tesla's stock price increased 50% in Q3 2023",
        summaries="The claim discusses Tesla's stock performance in the third quarter of 2023.",
        contextual="According to market data, Tesla (TSLA) opened Q3 2023 at $250 and closed at $270, representing an 8% increase. The 50% claim is inaccurate.",
        label="False",
        explanation="The actual stock price increase was only 8%, not 50% as claimed. Market data shows TSLA moved from $250 to $270 in Q3 2023."
    )
    
    formatted = format_finfact_sample(sample, output_format='text')
    
    print("\n[INSTRUCTION TEMPLATE]")
    print("-" * 40)
    print(FINFACT_INSTRUCTION_TEMPLATE)
    
    print("\n[INPUT FORMAT]")
    print("-" * 40)
    print(FINFACT_INPUT_TEMPLATE.format(
        claim=sample.claim,
        summaries=sample.summaries,
        contextual=sample.contextual
    ))
    
    print("\n[EXPECTED OUTPUT]")
    print("-" * 40)
    print(f"Prediction: 0. False")
    print(f"Explanation: {sample.explanation}")
    
    print("\n[FULL FORMATTED SAMPLE (Llama 3.1 format)]")
    print("-" * 40)
    print(formatted['text'][:1000] + "..." if len(formatted['text']) > 1000 else formatted['text'])


def dummy_check_finguard():
    """Print a sample FinGuard prompt for manual verification against Paper Table 2."""
    print("\n" + "="*80)
    print("DUMMY CHECK: FinGuard Sample (Task 2)")
    print("="*80)
    
    sample = FinGuardSample(
        text="BREAKING: Major bank announces unprecedented 500% dividend increase for shareholders. Experts warn of market manipulation as stock prices surge.",
        label="Fake",
        source="fake"
    )
    
    formatted = format_finguard_sample(sample, output_format='text')
    
    print("\n[INSTRUCTION TEMPLATE]")
    print("-" * 40)
    print(FINGUARD_INSTRUCTION_TEMPLATE)
    
    print("\n[INPUT FORMAT]")
    print("-" * 40)
    print(FINGUARD_INPUT_TEMPLATE.format(text=sample.text))
    
    print("\n[EXPECTED OUTPUT]")
    print("-" * 40)
    print("0. Fake")
    
    print("\n[FULL FORMATTED SAMPLE (Llama 3.1 format)]")
    print("-" * 40)
    print(formatted['text'])


def run_verification():
    """Run all dummy checks for manual verification."""
    dummy_check_finfact()
    dummy_check_finguard()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("""
Please compare the above outputs with Paper Table 2:

FINFACT (Task 1):
- Instruction should start with: "Task: Please determine whether the claim is 0. False, 1. True, or 2. Not Enough Information (NEI)..."
- Input should have: "Claim: [claim]. Claim summaries: [summaries]. Contextual information: [contextual]"
- Output should have: "Prediction: [0. False/1. True/2. NEI]" followed by "Explanation: [...]"

FINGUARD (Task 2):
- Instruction should be: "Task: Please determine whether the text is 0. Fake or 1. True. Answer directly without explanations."
- Input should have: "Text: [input text]"
- Output should be just: "0. Fake" or "1. True"
""")


# ==============================================================================
# MAIN PROCESSING PIPELINE
# ==============================================================================

def download_instructions():
    """Print instructions for downloading the datasets."""
    print("\n" + "="*80)
    print("STEP 1: DATASET ACQUISITION")
    print("="*80)
    print(f"""
Clone the required repositories:

1. FinFact Dataset:
   git clone {FINFACT_REPO_URL} {FINFACT_DIR}
   
   Expected structure after cloning:
   {FINFACT_DIR}/
   ├── data/
   │   ├── train.json (or similar)
   │   ├── dev.json / val.json
   │   └── test.json
   ├── README.md
   └── ...

2. FinGuard (Financial Truth Guard) Dataset:
   git clone {FINGUARD_REPO_URL} {FINGUARD_DIR}
   
   Expected structure after cloning:
   {FINGUARD_DIR}/
   ├── data/
   │   ├── real_news.json (or real/ folder)
   │   ├── fake_news.json (or fake/ folder)
   │   └── ...
   ├── README.md
   └── ...

After cloning, run this script again with --process flag.
""")


def process_datasets(output_format: str = 'text'):
    """Main processing pipeline using default paths."""
    return process_datasets_with_paths(FINFACT_DIR, FINGUARD_DIR, OUTPUT_DIR, output_format)


def process_datasets_with_paths(finfact_path: Path, finguard_path: Path, output_path: Path, output_format: str = 'text'):
    """Main processing pipeline."""
    print("\n" + "="*80)
    print("STEP 2: PROCESSING DATASETS")
    print("="*80)
    
    # Process FinFact
    print("\n[Processing FinFact]")
    finfact_samples = load_finfact_data(finfact_path)
    
    if finfact_samples:
        finfact_filtered = filter_finfact_samples(finfact_samples)
        finfact_train, finfact_val, finfact_test = split_data(finfact_filtered)
    else:
        finfact_train, finfact_val, finfact_test = [], [], []
        print("  Skipping FinFact (no data found)")
    
    # Process FinGuard
    print("\n[Processing FinGuard]")
    real_samples, fake_samples = load_finguard_data(finguard_path)
    
    if real_samples or fake_samples:
        finguard_samples = sample_finguard_data(real_samples, fake_samples)
        finguard_train, finguard_val, finguard_test = split_data(finguard_samples)
    else:
        finguard_train, finguard_val, finguard_test = [], [], []
        print("  Skipping FinGuard (no data found)")
    
    # Format and save
    print("\n[Formatting and Saving]")
    
    # Format FinFact samples
    finfact_train_formatted = [format_finfact_sample(s, output_format) for s in finfact_train]
    finfact_val_formatted = [format_finfact_sample(s, output_format) for s in finfact_val]
    finfact_test_formatted = [format_finfact_sample(s, output_format) for s in finfact_test]
    
    # Format FinGuard samples
    finguard_train_formatted = [format_finguard_sample(s, output_format) for s in finguard_train]
    finguard_val_formatted = [format_finguard_sample(s, output_format) for s in finguard_val]
    finguard_test_formatted = [format_finguard_sample(s, output_format) for s in finguard_test]
    
    # Save individual datasets
    save_jsonl(finfact_train_formatted, output_path / "finfact" / "train.jsonl")
    save_jsonl(finfact_val_formatted, output_path / "finfact" / "val.jsonl")
    save_jsonl(finfact_test_formatted, output_path / "finfact" / "test.jsonl")
    
    save_jsonl(finguard_train_formatted, output_path / "finguard" / "train.jsonl")
    save_jsonl(finguard_val_formatted, output_path / "finguard" / "val.jsonl")
    save_jsonl(finguard_test_formatted, output_path / "finguard" / "test.jsonl")
    
    # Save combined datasets (for multi-task training)
    combined_train = finfact_train_formatted + finguard_train_formatted
    combined_val = finfact_val_formatted + finguard_val_formatted
    combined_test = finfact_test_formatted + finguard_test_formatted
    
    random.seed(RANDOM_SEED)
    random.shuffle(combined_train)
    random.shuffle(combined_val)
    random.shuffle(combined_test)
    
    save_jsonl(combined_train, output_path / "combined" / "train.jsonl")
    save_jsonl(combined_val, output_path / "combined" / "val.jsonl")
    save_jsonl(combined_test, output_path / "combined" / "test.jsonl")
    
    # Generate and save statistics
    stats = generate_dataset_stats(
        finfact_train, finfact_val, finfact_test,
        finguard_train, finguard_val, finguard_test
    )
    
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics to {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"""
Dataset Statistics:
-------------------
FinFact:
  - Train: {stats['finfact']['train']}
  - Val:   {stats['finfact']['val']}
  - Test:  {stats['finfact']['test']}
  - Total: {stats['finfact']['total']}

FinGuard:
  - Train: {stats['finguard']['train']}
  - Val:   {stats['finguard']['val']}
  - Test:  {stats['finguard']['test']}
  - Total: {stats['finguard']['total']}

Combined (FMD):
  - Train: {stats['combined']['train']}
  - Val:   {stats['combined']['val']}
  - Test:  {stats['combined']['test']}
  - Total: {stats['combined']['total']}

Output Directory: {output_path.absolute()}
""")
    
    return stats


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FMD Data Preparation Script for FMDLlama Replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python prepare_fmd_data.py --download
  
  # Process datasets after downloading
  python prepare_fmd_data.py --process
  
  # Run verification checks
  python prepare_fmd_data.py --verify
  
  # Process with messages format (for chat fine-tuning)
  python prepare_fmd_data.py --process --format messages
        """
    )
    
    parser.add_argument('--download', action='store_true',
                       help='Show instructions for downloading datasets')
    parser.add_argument('--process', action='store_true',
                       help='Process the downloaded datasets')
    parser.add_argument('--verify', action='store_true',
                       help='Run dummy checks for template verification')
    parser.add_argument('--format', choices=['text', 'messages'], default='text',
                       help='Output format for fine-tuning (default: text)')
    parser.add_argument('--finfact-dir', type=Path, default=FINFACT_DIR,
                       help='Path to FinFact repository')
    parser.add_argument('--finguard-dir', type=Path, default=FINGUARD_DIR,
                       help='Path to FinGuard repository')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Use paths from args
    finfact_path = args.finfact_dir
    finguard_path = args.finguard_dir
    output_path = args.output_dir
    
    print("="*80)
    print("FMD (Financial Misinformation Detection) Data Preparation")
    print("="*80)
    print(f"FinFact Repository:  {FINFACT_REPO_URL}")
    print(f"FinGuard Repository: {FINGUARD_REPO_URL}")
    print(f"Output Directory:    {output_path}")
    
    if args.verify:
        run_verification()
    elif args.download:
        download_instructions()
    elif args.process:
        process_datasets_with_paths(finfact_path, finguard_path, output_path, output_format=args.format)
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "-"*80)
        print("Quick start: Run with --download to see how to get the datasets,")
        print("then --process to create the fine-tuning data.")


if __name__ == "__main__":
    main()
