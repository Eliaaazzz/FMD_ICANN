"""
DAG_Construct.py
构建统一RAG数据库，支持多特征、可扩展、可区分版本
"""
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# DATA_PATHS = {
#     "finfact": "../data/FinFact/finfact_train.json",
#     "finguard_fake": "../data/FinGuard/FinGuard_FAKE/Finance_FAKE_train.csv",
#     "finguard_true": "../data/FinGuard/FinGuard_TRUE/Finance_TRUE_train.csv"
# }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = {
    "finfact": os.path.join(BASE_DIR, "..", "data", "FinFact", "finfact_train.json"),
    "finguard_fake": os.path.join(BASE_DIR, "..", "data", "FinGuard", "FinGuard_FAKE", "Finance_FAKE_train.csv"),
    "finguard_true": os.path.join(BASE_DIR, "..", "data", "FinGuard", "FinGuard_TRUE", "Finance_TRUE_train.csv"),
}

# 输出文件夹（使用 BASE_DIR 确保相对 ICANN）
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "DAG"))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LLM_MODEL = "qwen-max"
EMBEDDING_MODEL = "text-embedding-v4"

# 每个数据集生成条数（-1 表示全量）
FINFACT_LIMIT = -1
FINGUARD_TRUE_LIMIT = -1
FINGUARD_FALSE_LIMIT = -1

# 断点重跑配置
# 想断点续跑：把 RESUME 设为 True，并固定 RUN_TAG（不要改动）
RESUME = True
RUN_TAG = "20260211_002927"  # 设为 None 会自动生成时间戳；想续跑请写固定字符串，例如 "run_20260210"

# 初始化API
client = OpenAI(
    api_key="sk-09964cbd8e0446879cac5bac49a87aad",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 新特征建议 =================
# 新增特征的修改方式：
# 1) 在 FEATURES 里加字段名
# 2) 在 doc 构建处给该字段赋值
# 3) 如果需要 LLM 提取，在 extract_metadata_via_llm 里加提示
FEATURES = [
    "id", "source_dataset", "content", "embedding", "entities", "events", "timestamp",
    "author", "rhetoric_pattern", "credibility_score", "evidence_snippets", "justification",
    "source_url", "label"
]

# ================== 特征提取函数 ==================
def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:2048]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

def extract_metadata_via_llm(text: str, need_time=False, need_author=False, need_rhetoric=False) -> Dict:
    prompt = f"""
    You are an expert financial data analyst. Analyze the following text and extract key information into a strict JSON format.
    Text: \"{text[:1000]}...\"
    Please extract:
    1. entities: list[str], companies, organizations, or key figures.   # 这里只是举例子，开放的
    2. events: list[str], core financial event types.
    """
    if need_time:
        prompt += '\n3. timestamp: string, infer YYYY-MM-DD or Null.'
    if need_author:
        prompt += '\n4. author: string, infer author/source or Null.'
    if need_rhetoric:
        prompt += '\n5. rhetoric_pattern: string, rhetorical/deceptive pattern or Null.'
    prompt += "\nOutput ONLY the JSON string."
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for financial misinformation detection."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        content = completion.choices[0].message.content
        content = content.replace("```json","").replace("```","").strip()
        return json.loads(content)
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        return {"entities": [], "events": [], "timestamp": None, "author": None, "rhetoric_pattern": None}

# ================== 数据处理 ==================
def _resolve_end_index(total_count: int, limit: int) -> int:
    if limit < 0:
        return total_count
    return min(limit, total_count)

def _load_checkpoint(checkpoint_file: str) -> Dict:
    if not os.path.exists(checkpoint_file):
        return {"finfact": 0, "finguard_true": 0, "finguard_false": 0}
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_checkpoint(checkpoint_file: str, checkpoint: Dict) -> None:
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def _update_checkpoint(checkpoint_file: str, updates: Dict) -> None:
    checkpoint = _load_checkpoint(checkpoint_file)
    checkpoint.update({k: v for k, v in updates.items() if v is not None})
    _save_checkpoint(checkpoint_file, checkpoint)

def process_finfact(filepath: str, limit: int, start_index: int, output_file: str, checkpoint_file: str) -> int:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return start_index
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    end_index = _resolve_end_index(len(data), limit)
    if start_index >= end_index:
        return start_index
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for item in tqdm(data[start_index:end_index], desc="FinFact", total=end_index - start_index):
            claim = item.get("claim", "")
            meta = extract_metadata_via_llm(claim, need_time=False, need_author=True, need_rhetoric=True)
            doc = {
                "id": str(uuid.uuid4()),
                "source_dataset": "FinFact",
                "content": claim,
                "embedding": get_embedding(claim),
                "entities": meta.get("entities", []),
                "events": meta.get("events", []),
                "timestamp": item.get("posted", None),
                "author": item.get("author", meta.get("author", None)),
                "rhetoric_pattern": meta.get("rhetoric_pattern", None) if item.get("label") == "false" else None,
                "credibility_score": 0.95 if item.get("label") == "true" else (0.1 if item.get("label") == "false" else 0.5),
                "evidence_snippets": [ev.get("sentence","") for ev in item.get("evidence",[])],
                "justification": item.get("justification", ""),
                "source_url": item.get("url", None),
                "label": item.get("label", "Unknown")
            }
            out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            start_index += 1
            _update_checkpoint(checkpoint_file, {"finfact": start_index})
    return start_index

def process_finguard(filepath: str, label_type: str, limit: int, start_index: int, output_file: str, checkpoint_file: str, checkpoint_key: str) -> int:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return start_index
    df = pd.read_csv(filepath)
    end_index = _resolve_end_index(len(df), limit)
    if start_index >= end_index:
        return start_index
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for _, row in tqdm(df.iloc[start_index:end_index].iterrows(), desc=f"FinGuard_{label_type}", total=end_index - start_index):
            text = row.get('text', '')
            meta = extract_metadata_via_llm(text, need_time=True, need_author=True, need_rhetoric=True)
            doc = {
                "id": str(uuid.uuid4()),
                "source_dataset": f"FinGuard_{label_type}",
                "content": text,
                "embedding": get_embedding(text),
                "entities": meta.get("entities", []),
                "events": meta.get("events", []),
                "timestamp": meta.get("timestamp", None),
                "author": meta.get("author", None),
                "rhetoric_pattern": meta.get("rhetoric_pattern", None) if label_type == "False" else None,
                "credibility_score": 0.9 if label_type == "True" else 0.1,
                "evidence_snippets": [],
                "justification": "",
                "source_url": None,
                "label": label_type
            }
            out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            start_index += 1
            _update_checkpoint(checkpoint_file, {checkpoint_key: start_index})
    return start_index

# ================== 主流程 ==================
def main():
    run_tag = RUN_TAG or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"dag_db_{run_tag}.jsonl")
    checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_{run_tag}.json")

    if not RESUME and os.path.exists(output_file):
        os.remove(output_file)
    if not RESUME and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    checkpoint = _load_checkpoint(checkpoint_file)
    finfact_start = checkpoint.get("finfact", 0) or 0
    finguard_true_start = checkpoint.get("finguard_true", 0) or 0
    finguard_false_start = checkpoint.get("finguard_false", 0) or 0

    finfact_start = process_finfact(
        DATA_PATHS["finfact"], FINFACT_LIMIT, finfact_start, output_file, checkpoint_file
    )
    _save_checkpoint(checkpoint_file, {"finfact": finfact_start, "finguard_true": finguard_true_start, "finguard_false": finguard_false_start})

    finguard_true_start = process_finguard(
        DATA_PATHS["finguard_true"], "True", FINGUARD_TRUE_LIMIT, finguard_true_start, output_file, checkpoint_file, "finguard_true"
    )
    _save_checkpoint(checkpoint_file, {"finfact": finfact_start, "finguard_true": finguard_true_start, "finguard_false": finguard_false_start})

    finguard_false_start = process_finguard(
        DATA_PATHS["finguard_fake"], "False", FINGUARD_FALSE_LIMIT, finguard_false_start, output_file, checkpoint_file, "finguard_false"
    )
    _save_checkpoint(checkpoint_file, {"finfact": finfact_start, "finguard_true": finguard_true_start, "finguard_false": finguard_false_start})

    print(f"数据库已生成，文件名: {output_file}")

if __name__ == "__main__":
    main()
