import os
import json
import uuid
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
# 数据集路径配置
DATA_PATHS = {
    "finfact": "../data/FinFact/finfact.json",
    "finguard_fake": "../data/FinGuard/Finance_FAKE.csv",
    "finguard_true": "../data/FinGuard/Finance_TRUE.csv"
}

# 输出文件路径
OUTPUT_KB_FILE = "../data/CATO_Knowledge_base/cato_knowledge_base_train.jsonl"

# 模型名称配置
LLM_MODEL = "qwen-max"  # 用于语义提取的大语言模型
EMBEDDING_MODEL = "text-embedding-v4"  # 用于向量化的模型

# 训练集比例配置 (0.7 表示取前 70%)
TRAIN_RATIO = 0.7

# 调试模式：True 时仅处理少量数据用于测试代码，False 时跑全量数据
DEBUG_MODE = True
DEBUG_LIMIT = 2

# ================= 初始化客户端 =================
# 请确保环境变量 DASHSCOPE_API_KEY 已设置，或直接在此处填入 key
client = OpenAI(
    api_key="sk-6234f2144f4946fa81cbfaf6e382c3a0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class KnowledgeBaseBuilder:
    def __init__(self, train_ratio: float = 0.7):
        """
        初始化构建器
        :param train_ratio: 训练集切分比例，默认 0.7
        """
        self.kb_data = []
        self.train_ratio = train_ratio

    def _get_embedding(self, text: str) -> List[float]:
        """
        调用 Embedding 模型将文本转化为向量
        """
        try:
            # 简单截断防止超过 token 限制 (根据模型实际限制调整)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:2048]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"向量化错误 (Embedding Error): {e}")
            return []

    def _extract_metadata_via_llm(self, text: str, extract_time: bool = False, analyze_rhetoric: bool = False) -> Dict:
        """
        利用 LLM (qwen-max) 提取关键元数据。
        Prompt 全部使用英文以适配英文数据集，提高准确率。
        """
        # 基础 Prompt：提取实体和事件
        prompt = f"""
        You are an expert financial data analyst. Analyze the following text and extract key information into a strict JSON format.

        Text: "{text[:1000]}..."

        Please extract the following fields:
        1. "entities": list[str], specific companies, organizations, or key figures mentioned (e.g., ["Tesla", "Elon Musk", "SEC"]).
        2. "events": list[str], core financial event types (e.g., ["Merger", "Stock Plunge", "Regulatory Fine"]).
        """

        # 针对 FinGuard 缺失时间的情况，要求 LLM 推断
        if extract_time:
            prompt += '\n3. "timestamp": string, infer the specific date of the event (format YYYY-MM-DD). If not explicitly stated, infer from context. If impossible, return "Unknown".'

        # 针对虚假新闻 (False 类)，要求分析修辞/欺诈模式
        if analyze_rhetoric:
            prompt += '\n4. "rhetoric_pattern": string, identify the rhetorical strategy or deceptive pattern used (e.g., "Emotional Manipulation", "False Attribution", "Fabricated Statistics").'

        prompt += "\n\nOutput ONLY the JSON string. Do not include Markdown formatting."

        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant for financial misinformation detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # 低温度确保输出稳定
            )
            content = completion.choices[0].message.content
            # 清理可能的 markdown 标记
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"LLM 提取错误 (LLM Extraction Error): {e}")
            # 返回空值防止程序中断
            return {"entities": [], "events": [], "timestamp": "Unknown", "rhetoric_pattern": "None"}

    def process_finfact(self, filepath: str):
        """
        处理 FinFact 数据集 (JSON 列表结构)
        """
        if not os.path.exists(filepath):
            print(f"文件未找到: {filepath}")
            return

        print(f"正在加载 FinFact 数据: {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        total_count = len(full_data)
        # 计算切分点
        split_index = int(total_count * self.train_ratio)
        train_data = full_data[:split_index]

        print(f"FinFact 总样本数: {total_count}, 训练集样本数 (前 {self.train_ratio * 100}%): {len(train_data)}")

        processed_count = 0
        for item in tqdm(train_data, desc="Processing FinFact"):
            if DEBUG_MODE and processed_count >= DEBUG_LIMIT: break

            # 1. 提取 claim (核心文本)
            claim = item.get("claim", "")
            if not claim: continue

            # 2. 调用 LLM 提取实体 (FinFact 自带时间 posted，通常不需要 LLM 猜时间)
            meta = self._extract_metadata_via_llm(claim, extract_time=False)

            # 3. 向量化
            vector = self._get_embedding(claim)

            # 4. 构建知识库条目
            doc = {
                "id": str(uuid.uuid4()),
                "source_dataset": "FinFact",
                "content": claim,
                "embedding": vector,
                "entities": meta.get("entities", []),
                "events": meta.get("events", []),
                "timestamp": item.get("posted", "Unknown"),  # 优先使用原数据时间
                "label": item.get("label", "Unknown"),
                "credibility_score": 0.95,  # FinFact 经过专家核查，可信度高
                # 聚合 evidence 列表为字符串列表
                "evidence_snippets": [ev.get("sentence", "") for ev in item.get("evidence", [])],
                "justification": item.get("justification", ""),  # 核心推理逻辑
                "rhetoric_pattern": None
            }
            self.kb_data.append(doc)
            processed_count += 1

    def process_finguard(self, filepath: str, label_type: str):
        """
        处理 FinGuard 数据集 (CSV 结构)
        """
        if not os.path.exists(filepath):
            print(f"文件未找到: {filepath}")
            return

        print(f"正在加载 FinGuard ({label_type}) 数据: {filepath}...")
        df = pd.read_csv(filepath)

        total_count = len(df)
        # 计算切分点
        split_index = int(total_count * self.train_ratio)
        # 切片获取前 X% 的数据
        train_df = df.iloc[:split_index]

        print(f"FinGuard ({label_type}) 总样本数: {total_count}, 训练集样本数: {len(train_df)}")

        processed_count = 0
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Processing {label_type}"):
            if DEBUG_MODE and processed_count >= DEBUG_LIMIT: break

            # 假设 CSV 文本列名为 'text'，请根据实际情况修改
            full_text = row.get('text', '')
            if not isinstance(full_text, str) or len(full_text) < 10: continue

            # 文本切片：取前 1000 字符作为主要索引内容（可视情况调整）
            chunk_text = full_text[:1000]

            # 1. 调用 LLM 提取 (FinGuard 缺时间，Fake 类需要提取修辞模式)
            is_fake = (label_type == "False")
            meta = self._extract_metadata_via_llm(
                chunk_text,
                extract_time=True,
                analyze_rhetoric=is_fake
            )

            # 2. 向量化
            vector = self._get_embedding(chunk_text)

            # 3. 构建知识库条目
            doc = {
                "id": str(uuid.uuid4()),
                "source_dataset": f"FinGuard_{label_type}",
                "content": chunk_text,
                "embedding": vector,
                "entities": meta.get("entities", []),
                "events": meta.get("events", []),
                "timestamp": meta.get("timestamp", "Unknown"),  # LLM 推断的时间
                "label": label_type,
                # True 新闻作为事实背景(0.9)，False 新闻作为反面教材(0.1，或者标记为负样本)
                "credibility_score": 0.9 if label_type == "True" else 0.1,
                "evidence_snippets": [],
                "justification": "",
                "rhetoric_pattern": meta.get("rhetoric_pattern", None)  # 仅 Fake 类有此字段
            }
            self.kb_data.append(doc)
            processed_count += 1

    def save_to_jsonl(self):
        """
        保存结果到 JSONL 文件
        """
        print(f"正在保存 {len(self.kb_data)} 条数据到 {OUTPUT_KB_FILE}...")
        with open(OUTPUT_KB_FILE, 'w', encoding='utf-8') as f:
            for entry in self.kb_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print("保存完成。")


# ================= 主执行流程 =================
if __name__ == "__main__":
    # 实例化构建器，传入自定义的训练集比例
    builder = KnowledgeBaseBuilder(train_ratio=TRAIN_RATIO)

    # 1. 处理 FinFact (逻辑推理核心)
    builder.process_finfact(DATA_PATHS["finfact"])

    # 2. 处理 FinGuard False (虚假模式库)
    builder.process_finguard(DATA_PATHS["finguard_fake"], label_type="False")

    # 3. 处理 FinGuard True (世界知识补充)
    builder.process_finguard(DATA_PATHS["finguard_true"], label_type="True")

    # 4. 持久化保存
    builder.save_to_jsonl()