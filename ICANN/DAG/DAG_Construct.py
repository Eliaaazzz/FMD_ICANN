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

# ================= 配置区域 =================
# DATA_PATHS = {
#     "finfact": "../data/FinFact/finfact_train.json",
#     "finguard_fake": "../data/FinGuard/FinGuard_FAKE/Finance_FAKE_train.csv",
#     "finguard_true": "../data/FinGuard/FinGuard_TRUE/Finance_TRUE_train.csv"
# }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = {
    "finfact": os.path.join(BASE_DIR, "..", "data", "FinFact", "finfact_train1.json"),
    "finguard_fake": os.path.join(BASE_DIR, "..", "data", "FinGuard", "FinGuard_FAKE", "Finance_FAKE_train1.csv"),
    "finguard_true": os.path.join(BASE_DIR, "..", "data", "FinGuard", "FinGuard_TRUE", "Finance_TRUE_train.csv"),
}

# 输出文件夹
OUTPUT_DIR = "../data/DAG"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 模型配置
LLM_MODEL = "qwen-max"
EMBEDDING_MODEL = "text-embedding-v4"

# 生成数据条数（每类5条）
DEBUG_LIMIT = 5

# 初始化API
client = OpenAI(
    api_key="sk-6234f2144f4946fa81cbfaf6e382c3a0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 新特征建议 =================
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
    1. entities: list[str], companies, organizations, or key figures.
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
def process_finfact(filepath: str, limit: int) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for item in data[:limit]:
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
        result.append(doc)
    return result

def process_finguard(filepath: str, label_type: str, limit: int) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    df = pd.read_csv(filepath)
    result = []
    for _, row in df.iloc[:limit].iterrows():
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
        result.append(doc)
    return result

# ================== 主流程 ==================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"dag_db_{timestamp}.jsonl")
    all_docs = []
    all_docs.extend(process_finfact(DATA_PATHS["finfact"], DEBUG_LIMIT))
    all_docs.extend(process_finguard(DATA_PATHS["finguard_true"], "True", DEBUG_LIMIT))
    all_docs.extend(process_finguard(DATA_PATHS["finguard_fake"], "False", DEBUG_LIMIT))
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"数据库已生成，共{len(all_docs)}条，文件名: {output_file}")

if __name__ == "__main__":
    main()
