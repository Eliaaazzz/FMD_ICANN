import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
import logging

# === 引入 CATO 模块 ===
from Retrieval_Planner import HybridRetriever, MetaCognitivePlanner
from Toolset import FinancialFactCheckingTools
from Router import DynamicDAGScheduler
from Fusion_Engine import EvidenceFusionLayer

# ==========================================
# 1. 配置区域 (User Configuration)
# ==========================================

API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 路径配置（基于脚本所在目录，不受 cwd 影响）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(_BASE_DIR, "..", "data", "DAG", "Final_Combined_Data.jsonl")
FINFACT_PATH = os.path.join(_BASE_DIR, "..", "data", "Split_data", "val", "finfact_val.json")
FINGUARD_FAKE_PATH = os.path.join(_BASE_DIR, "..", "data", "Split_data", "val", "Finance_FAKE_val.csv")
FINGUARD_TRUE_PATH = os.path.join(_BASE_DIR, "..", "data", "Split_data", "val", "Finance_TRUE_val.csv")
OUTPUT_FILE = os.path.join(_BASE_DIR, "results", "evaluation_results.json")

# 测试参数
TEST_PARAMS = {
    "FinFact": 5,  # 建议 FinFact 多测几条以验证 NEI
    "FinGuard_FAKE": 5,
    "FinGuard_TRUE": 5
}

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ==========================================
# 2. 系统初始化 (System Init)
# ==========================================

class CATOSystem:
    def __init__(self):
        logger.info("Initializing CATO System...")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.corpus = self._load_knowledge_base()
        self.retriever = HybridRetriever(corpus=self.corpus, client=self.client)
        self.planner = MetaCognitivePlanner(client=self.client, model="qwen-max")
        self.toolset = FinancialFactCheckingTools(client=self.client, model="qwen-max")
        self.scheduler = DynamicDAGScheduler(toolset=self.toolset)
        self.fusion = EvidenceFusionLayer(client=self.client, model="qwen-max")

    def _load_knowledge_base(self):
        logger.info(f"Loading Knowledge Base from {KNOWLEDGE_BASE_PATH}...")
        corpus = []
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            return []
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        corpus.append(json.loads(line))
                    except:
                        continue
        logger.info(f"Loaded {len(corpus)} documents.")
        return corpus

    def run_single_inference(self, text):
        try:
            evidence = self.retriever.search(text, top_k=3)
            plan_list = self.planner.generate_plan(text, evidence)
            # 兼容 Planner 返回 list 或 json 的情况
            if isinstance(plan_list, list):
                plan_json = {"tools": plan_list, "dependencies": []}
            else:
                plan_json = plan_list  # 已经是json

            result_context = self.scheduler.execute(plan_json, text, evidence)
            final_report = self.fusion.generate_final_report(result_context)
            return final_report
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"final_label": "Error", "risk_level": "Unknown"}


# ==========================================
# 3. 核心修正：3分类映射逻辑
# ==========================================

def load_finfact_data(limit):
    """
    [修正] 加载 FinFact 数据并映射为 3 类标签
    0: Real (True)
    1: Fake (False)
    2: NEI (Not Enough Info / Unverified / Mixture)
    """
    data = []
    if not os.path.exists(FINFACT_PATH):
        logger.warning(f"File not found: {FINFACT_PATH}")
        return data

    try:
        with open(FINFACT_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        if limit != -1: full_data = full_data[:limit]

        for item in full_data:
            text = item.get('claim') or item.get('text')
            raw_label = str(item.get('label', '')).lower()

            # --- 3分类映射逻辑 ---
            if raw_label in ['true', 'mostly true', 'supports', 'supported']:
                label = 0  # Real
            elif raw_label in ['false', 'mostly false', 'refutes', 'refuted']:
                label = 1  # Fake
            else:
                # 包括: NEI, Mixture, Unverified, null, etc.
                label = 2  # NEI

            data.append({"text": text, "label": label, "source": "FinFact", "raw_label": raw_label})
    except Exception as e:
        logger.error(f"Error loading FinFact: {e}")
    return data


def load_finguard_data(fake_limit, true_limit):
    """
    [保持] FinGuard 是纯二分类数据集
    0: Real
    1: Fake
    """
    data = []
    # Load FAKE (Label 1)
    if os.path.exists(FINGUARD_FAKE_PATH):
        df = pd.read_csv(FINGUARD_FAKE_PATH)
        if fake_limit != -1: df = df.head(fake_limit)
        for _, row in df.iterrows():
            text = row.get('text') or row.get('statement') or row.iloc[0]
            data.append({"text": str(text), "label": 1, "source": "FinGuard_FAKE"})

    # Load TRUE (Label 0)
    if os.path.exists(FINGUARD_TRUE_PATH):
        df = pd.read_csv(FINGUARD_TRUE_PATH)
        if true_limit != -1: df = df.head(true_limit)
        for _, row in df.iterrows():
            text = row.get('text') or row.get('statement') or row.iloc[0]
            data.append({"text": str(text), "label": 0, "source": "FinGuard_TRUE"})

    return data


def map_prediction_to_label(report, dataset_type="3_class"):
    """
    [修正] 将 LLM 输出映射为数字标签
    """
    final_label = str(report.get('final_label', '')).lower()
    risk_level = str(report.get('risk_level', '')).lower()

    # 1. 尝试直接通过 Label 文本判断
    if "real" in final_label or "true" in final_label:
        pred = 0
    elif "fake" in final_label:
        pred = 1
    elif "unverified" in final_label or "misleading" in final_label or "mixture" in final_label:
        pred = 2
    else:
        # 兜底：如果 Label 没写清楚，看风险等级
        if risk_level in ['critical', 'high']:
            pred = 1  # Fake
        elif risk_level == 'low':
            pred = 0  # Real
        else:
            pred = 2  # Medium/Unknown -> NEI

    # 如果是二分类数据集(FinGuard)，强制把 NEI(2) 归为 Fake(1) 或者根据业务需求处理
    # 这里为了指标计算方便，FinGuard 我们通常只看 0 和 1
    if dataset_type == "2_class":
        if pred == 2: pred = 1  # 视“无法验证”为“高风险/假” (或者你可以改为 1)

    return pred


def calculate_metrics(y_true, y_pred, dataset_name, labels=None, target_names=None):
    """
    [修正] 使用 classification_report 计算多分类指标
    """
    print(f"\n{'=' * 20} {dataset_name} Evaluation {'=' * 20}")
    if not y_true:
        print("No data.")
        return

    # 打印混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", cm)

    # 打印详细指标 (P/R/F1)
    # zero_division=0 防止除以零报错
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print("\nDetailed Metrics:\n", report)
    print("-" * 60)


# ==========================================
# 4. 主程序
# ==========================================

def main():
    cato = CATOSystem()

    # 1. 加载数据
    logger.info("Loading Datasets...")
    finfact_data = load_finfact_data(TEST_PARAMS["FinFact"])
    finguard_data = load_finguard_data(TEST_PARAMS["FinGuard_FAKE"], TEST_PARAMS["FinGuard_TRUE"])

    all_data = finfact_data + finguard_data
    logger.info(f"Total Test Samples: {len(all_data)}")

    results_log = []

    # 容器
    y_true_finfact, y_pred_finfact = [], []
    y_true_finguard, y_pred_finguard = [], []

    print("\n>>> Starting Inference...")
    for item in tqdm(all_data):
        text = item['text']
        true_label = item['label']
        source = item['source']

        # 推理
        report = cato.run_single_inference(text)

        # 映射预测结果
        if source == "FinFact":
            # FinFact 使用 3 分类 (0, 1, 2)
            pred_label = map_prediction_to_label(report, dataset_type="3_class")
            y_true_finfact.append(true_label)
            y_pred_finfact.append(pred_label)
        else:
            # FinGuard 使用 2 分类 (0, 1)
            pred_label = map_prediction_to_label(report, dataset_type="2_class")
            y_true_finguard.append(true_label)
            y_pred_finguard.append(pred_label)

        # 记录
        results_log.append({
            "source": source,
            "text": text[:50] + "...",
            "true_label": true_label,
            "pred_label": pred_label,
            "is_correct": true_label == pred_label,
            "raw_output": report
        })

    # 2. 计算指标 (区分对待)

    # === FinFact (3分类) ===
    # 0=Real, 1=Fake, 2=NEI
    calculate_metrics(
        y_true_finfact,
        y_pred_finfact,
        "FinFact (3-Class)",
        labels=[0, 1, 2],
        target_names=["Real", "Fake", "NEI"]
    )

    # === FinGuard (2分类) ===
    # 0=Real, 1=Fake
    calculate_metrics(
        y_true_finguard,
        y_pred_finguard,
        "FinGuard (Binary)",
        labels=[0, 1],
        target_names=["Real", "Fake"]
    )

    # 3. 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()