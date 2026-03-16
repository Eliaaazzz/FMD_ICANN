# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=1.25,<2.0",
#   "pandas",
#   "faiss-cpu",
#   "rank_bm25",
#   "openai",
#   "tqdm",
# ]
# ///

import os
import re
import json
import time
import logging
import random
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from Retrieval_Planner import HybridRetriever, MetaCognitivePlanner
from Toolset import FinancialFactCheckingTools
from Router_1 import DynamicDAGScheduler
from Fusion_Engine import DualStageFusionEngine


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINFACT_PATH = os.path.join(PROJECT_ROOT, "data", "Split_data", "val", "finfact_val.json")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

# ==========================================
# 1. 配置区域 (User Configuration)
# ==========================================

API_KEYS = [
    "sk-50faa0a25abf4340b3398af9ffea6168",
    "sk-1cb274266c2341d3be77f5b8fb0cea25",
    "sk-3e329ef6c7d24104ac8d1470418a970e",
    "sk-09964cbd8e0446879cac5bac49a87aad",
    "sk-a5866e555f2448b3a166ba2c7b252703",
    "sk-5449638cc6614016b9964f8d961294ee",
    "sk-81632092914e4b189ce3c20729072a83",
    "sk-f3d2b0beefc4450db3e58f7c2b1c2237",
]

TEST_PARAMS = {
    "FinFact": 10,
}

# 并发配置：外层同时处理的新闻条数（建议设置为 API Key 数量的 1~1.5 倍）
MAX_CONCURRENT_ITEMS = 8

# Unknown 解决配置
UNKNOWN_RETRY_TIMES = 2
UNKNOWN_RETRY_DELAY_SECONDS = 0.8
UNKNOWN_FALLBACK_LABEL = "False"

DEFAULT_PROMPTS = [
    "dual_track_verifier",
    "weighted_evidence_scorer",
    "cross_check_simulator",
    "tribunal_judgment",
    "editorial_board_vote",
]

# Prompt 选择模式：True=自动使用 PROMPT_TEMPLATES 中全部模板；False=只跑 DEFAULT_PROMPTS
RUN_ALL_PROMPTS = True  # 暂时关闭：先跑不含 {sci_digest} 的 5 个原始 prompt，排除缺失字段问题

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-max"


PROMPT_TEMPLATES = {
    "dual_track_verifier": {
        "system": """You are a dual-track verification system that performs both quantitative scoring and qualitative cross-referencing in parallel, then synthesizes results for final judgment.
Track A: Quantitative Scoring (0-30 scale)
Track B: Qualitative Cross-Reference (Pass/Fail checks)""",
        "user": """Execute dual-track verification on the following claim.

[CLAIM]
{claim}

[REFERENCE MATERIALS]
Summary: {justification}
Evidence: {evidence}

══════════════════════════════════════════
[TRACK A: QUANTITATIVE SCORING]
A1. Factual Accuracy (0-10): ___
A2. Evidence Alignment (0-10): ___
A3. Logical Consistency (0-10): ___
Track A Total: ___/30

══════════════════════════════════════════
[TRACK B: QUALITATIVE CROSS-REFERENCE]
B1. Core claim exists in evidence? □ PASS □ FAIL
B2. No contradictions detected? □ PASS □ FAIL
B3. Context preserved correctly? □ PASS □ FAIL
Track B Result: ___ of 3 PASS

══════════════════════════════════════════
[SYNTHESIS DECISION MATRIX]
| Track A Score | Track B Passes | Final Decision |
|---------------|----------------|----------------|
| >= 15         | >= 2           | True           |
| >= 18         | >= 1           | True           |
| < 15          | 3              | True           |
| Otherwise     | -              | False          |

Final Output: Prediction: True or Prediction: False""",
    },
    "weighted_evidence_scorer": {
        "system": """你是一个基于证据权重的评分系统。你会对声明的可信度要素进行打分（0-10分），总分低于20分（满分30）将被标记为False。""",
        "user": """请对以下声明进行基于证据的打分评估。

【声明】
{claim}

【证据材料】
{justification}
{evidence}

【评分项】
A. 证据覆盖度 (0-10): 0=无直接证据，10=证据完全覆盖声明的所有细节
B. 一致性 (0-10): 0=声明与证据矛盾，10=声明与证据高度一致
C. 语境准确性 (0-10): 0=严重断章取义，10=完全忠实于原意

请计算总分。
Decision Rule: Total Score >= 20 -> True; Total Score < 20 -> False.
输出格式：Prediction: True 或 Prediction: False""",
    },
    "cross_check_simulator": {
        "system": """You are a research assistant simulating a cross-referencing process.
You use the provided "Justification" and "Evidence" as your ground truth knowledge base to verify the "Claim".""",
        "user": """Verify the claim by cross-referencing it against the provided ground truth.

[Target Claim]
{claim}

[Ground Truth Knowledge Base]
{justification}
{evidence}

[Simulation]
- Initial Check: Does the claim exist in the Knowledge Base?
- Detail Verification: Do specific numbers, dates, and entities match exactly?
- Conflict Detection: Is there any statement in the Knowledge Base that directly contradicts the claim?

Verdict:
Prediction: True or Prediction: False""",
    },
    "tribunal_judgment": {
        "system": """你是一个事实审判庭系统，采用三阶段审判流程：
阶段一：证据审查（交叉验证）
阶段二：专家评分（量化评估）
阶段三：陪审投票（多角色决策）
三阶段综合判断确保结论的可靠性。""",
        "user": """开始对以下声明进行三阶段审判。

【被审声明】
{claim}

【案卷材料】
论证：{justification}
证据：{evidence}

═══════════════════════════════════════
【阶段一：证据审查】
- 声明核心内容是否存在于证据中？ □是 □否
- 具体细节（数字/日期/名称）是否匹配？ □是 □否
- 是否发现矛盾信息？ □无矛盾 □有矛盾

═══════════════════════════════════════
【阶段二：专家评分】
- 事实准确度: ___/10
- 证据充分度: ___/10
- 逻辑严密度: ___/10
总分: ___/30

═══════════════════════════════════════
【阶段三：陪审投票】
- 首席核查员: □支持True □支持False
- 证据分析师: □支持True □支持False
- 逻辑审计师: □支持True □支持False

═══════════════════════════════════════
【最终裁决】
综合三阶段结果：
- 阶段一无重大问题 + 阶段二总分>=15 + 阶段三多数支持True → Prediction: True
- 否则 → Prediction: False""",
    },
    "editorial_board_vote": {
        "system": """你模拟一个拥有三位资深成员的事实核查委员会：
1. 首席核查员（关注证据链的完整性）
2. 领域专家（关注术语和概念的准确性）
3. 逻辑分析师（关注推理过程的严密性）
你们需要投票决定该声明是否属实。""",
        "user": """委员会请就位，对以下声明进行审核投票。

【声明】
{claim}

【案卷材料】
论证：{justification}
证据：{evidence}

【委员会讨论】
- 首席核查员意见：...
- 领域专家意见：...
- 逻辑分析师意见：...

【最终投票结果】
如果至少两票认为属实(True)，则判定为真。
输出格式：Prediction: True 或 Prediction: False""",
    },


    # 优化6: 增强版交叉验证（更系统化的检查）
    "systematic_cross_validator": {
        "system": """You are an advanced cross-validation system with a systematic 5-step verification protocol.
You treat the justification and evidence as the ground truth and systematically verify every aspect of the claim.""",
        "user": """Execute the systematic cross-validation protocol.

[TARGET CLAIM]
{claim}

[GROUND TRUTH DATABASE]
Justification: {justification}
Evidence: {evidence}

[5-STEP VERIFICATION PROTOCOL]

Step 1 - Existence Check:
□ Does the core assertion exist in the ground truth?
□ Result: PASS / FAIL

Step 2 - Precision Verification:
□ Do specific numbers, percentages, dates match exactly?
□ Are entity names and titles accurate?
□ Result: PASS / FAIL

Step 3 - Logical Alignment:
□ Is the claim's conclusion supported by the justification's reasoning?
□ Are there any logical gaps or leaps?
□ Result: PASS / FAIL

Step 4 - Contradiction Scan:
□ Is there any statement in the ground truth that contradicts the claim?
□ Result: PASS (no contradiction) / FAIL (contradiction found)

Step 5 - Context Integrity:
□ Is the claim presented with proper context?
□ Does it faithfully represent the source material?
□ Result: PASS / FAIL

[FINAL VERDICT]
If >= 4 steps PASS → Prediction: True
If < 4 steps PASS → Prediction: False"""
    },

    "detailed_scoring_validator": {
        "system": """你是一个精细化评分验证系统。
你会对声明的每个关键维度进行交叉验证并打分，确保评估的全面性和准确性。""",
        "user": """请对以下声明进行精细化评分验证。

【待验证声明】
{claim}

【验证知识库】
论证：{justification}
证据：{evidence}

【精细化评分验证】
1. 事实匹配度 (0-10分)
   - 声明中的具体事实是否与知识库一致？
   - 数字、日期、实体名称是否准确？
   评分: ___

2. 论证支持度 (0-10分)
   - 论证材料是否支持该声明？
   - 是否存在直接矛盾？
   评分: ___

3. 证据覆盖度 (0-10分)
   - 声明的核心观点是否有证据支撑？
   - 证据来源是否可靠？
   评分: ___

4. 语境准确度 (0-10分)
   - 声明是否在正确语境下呈现？
   - 是否存在断章取义或误导？
   评分: ___

【决策阈值】总分 >= 18 → True; 总分 < 18 → False
输出格式：Prediction: True 或 Prediction: False"""
    },
    "enhanced_weighted_scorer": {
        "system": """你是一个增强版证据权重评分系统。
你会从5个核心维度对声明进行评分，每个维度0-10分，总分50分。
使用更严格的阈值确保判断的准确性。""",
        "user": """请对以下声明进行五维度评分评估。

【声明】
{claim}

【参考材料】
论证：{justification}
证据：{evidence}

【五维度评分】
A. 事实准确性 (0-10): 声明中的事实陈述是否准确无误？
B. 证据支持度 (0-10): 提供的证据是否充分支持该声明？
C. 逻辑一致性 (0-10): 声明与论证之间是否逻辑自洽？
D. 语境完整性 (0-10): 声明是否在完整语境下呈现？
E. 来源可信度 (0-10): 证据来源是否权威可靠？

【评分汇总】
A + B + C + D + E = 总分

【决策规则】
总分 >= 25 → Prediction: True
总分 < 25 → Prediction: False"""
    },
        "expert_panel_scoring": {
        "system": """你是一个由三位专家组成的评分委员会系统。
每位专家从自己的专业角度对声明进行0-10分评估，最终根据总分决定真实性。
- 证据分析师：评估证据的充分性和可靠性
- 逻辑审计师：评估推理过程的严密性
- 语境专家：评估声明是否被正确理解和呈现""",
        "user": """请三位专家分别对以下声明进行评分。

【声明】
{claim}


【论证材料】
{justification}

【证据】
{evidence}

【专家评分】
📊 证据分析师评分 (0-10): 证据是否充分支持或反驳声明？来源是否可靠？
📊 逻辑审计师评分 (0-10): 从证据到结论的推理是否严密？是否存在逻辑谬误？
📊 语境专家评分 (0-10): 声明是否在正确语境下被理解？是否断章取义？

【决策规则】
总分 >= 15 → True（声明属实）
总分 < 15 → False（声明虚假/误导）

输出格式：Prediction: True 或 Prediction: False"""
    },
        # 融合4: 多角色 + 交叉验证协议
    "committee_cross_reference": {
        "system": """You are a fact-checking committee that uses cross-referencing methodology.
Three committee members independently cross-reference the claim against the evidence base:
- Senior Fact-Checker: Focus on factual accuracy
- Research Analyst: Focus on source credibility and evidence quality
- Editorial Director: Focus on context and potential misleading elements""",
        "user": """Committee members, please cross-reference and evaluate the following claim.

[CLAIM TO EVALUATE]
{claim}

[REFERENCE DATABASE]
Justification: {justification}
Evidence: {evidence}

[COMMITTEE CROSS-REFERENCE ANALYSIS]

👤 Senior Fact-Checker:
- Cross-reference: Does each fact in the claim appear in the reference database?
- Accuracy check: Are all details (numbers, dates, names) correct?
- Verdict: ___

👤 Research Analyst:
- Source quality: Are the evidence sources credible and verifiable?
- Coverage: Does the evidence sufficiently support the claim?
- Verdict: ___

👤 Editorial Director:
- Context check: Is the claim presented without misleading omissions?
- Interpretation: Is the claim a fair representation of the source material?
- Verdict: ___

[COMMITTEE DECISION]
If majority (>=2) approve → Prediction: True
Otherwise → Prediction: False"""
    },


}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class MultiKeyClient:
    def __init__(self, api_keys, base_url):
        valid_keys = [k for k in api_keys if k and k.startswith("sk-")]
        if not valid_keys:
            raise ValueError("No valid API keys found. Set DASHSCOPE_API_KEYS or keep keys in main.py")
        self.clients = [OpenAI(api_key=key, base_url=base_url) for key in valid_keys]
        logger.info("Initialized MultiKeyClient with %s API keys", len(valid_keys))

    @property
    def chat(self):
        return random.choice(self.clients).chat

    @property
    def embeddings(self):
        return random.choice(self.clients).embeddings


def resolve_api_keys():
    env_keys = [k.strip() for k in os.getenv("DASHSCOPE_API_KEYS", "").split(",") if k.strip()]
    if env_keys:
        return env_keys

    return [k for k in API_KEYS if k and k.startswith("sk-")]


def load_knowledge_base(path):
    corpus = []
    if not os.path.exists(path):
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    corpus.append(json.loads(line))
                except Exception:
                    continue
    return corpus


def load_finfact_data(limit=100):
    data = []
    if not os.path.exists(FINFACT_PATH):
        raise FileNotFoundError(f"FinFact file not found: {FINFACT_PATH}")

    with open(FINFACT_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    for item in full_data:
        claim = item.get("claim") or item.get("text")
        raw_label = str(item.get("label", "")).lower()

        if raw_label in ["true", "mostly true", "supports", "supported"]:
            label = "True"
        elif raw_label in ["false", "mostly false", "refutes", "refuted"]:
            label = "False"
        else:
            continue

        data.append(
            {
                "text": str(claim),
                "label": label,
                "raw_label": raw_label,
                "source": "FinFact",
            }
        )

    if limit != -1:
        data = data[:limit]

    return data


class PromptExperimentSystem:
    def __init__(self, model=DEFAULT_MODEL):
        api_keys = resolve_api_keys()
        self.client = MultiKeyClient(api_keys=api_keys, base_url=BASE_URL)
        self.model = model

        kb_path = os.path.join(PROJECT_ROOT, "data", "DAG", "Final_Combined_Data.jsonl")
        logger.info("Loading knowledge base: %s", kb_path)
        self.corpus = load_knowledge_base(kb_path)
        logger.info("Loaded %s knowledge entries", len(self.corpus))

        self.retriever = HybridRetriever(corpus=self.corpus, client=self.client)
        self.planner = MetaCognitivePlanner(client=self.client, model=self.model)
        self.toolset = FinancialFactCheckingTools(client=self.client, model=self.model)
        self.scheduler = DynamicDAGScheduler(toolset=self.toolset)
        self.fusion = None

    def set_fusion_prompt(self, prompt_name):
        self.fusion = DualStageFusionEngine(
            llm_client=self.client,
            model=self.model,
            finfact_prompt_name=prompt_name,
        )
        self.fusion.FINFACT_PROMPTS.update(PROMPT_TEMPLATES)

    def run_single_inference(self, text, source="FinFact"):
        evidence = self.retriever.search(text, top_k=3)
        plan_list = self.planner.generate_plan(text, evidence)

        if isinstance(plan_list, list):
            plan_json = {"tools": plan_list, "dependencies": []}
        else:
            plan_json = plan_list

        result_context = self.scheduler.execute(plan_json, text, evidence)
        fusion_result = self.fusion.fuse(result_context, dataset_source=source)

        # 附加 pipeline 中间过程信息
        fusion_result["_pipeline_trace"] = {
            "retrieved_evidence": evidence,
            "plan": plan_json,
        }
        return fusion_result


def extract_prediction(report):
    final_label = str(report.get("final_label", "")).lower()
    if "real" in final_label or "true" in final_label:
        return "True"
    if "fake" in final_label or "false" in final_label:
        return "False"

    explanation = str(report.get("explanation_path", ""))
    match = re.search(r"Prediction:\s*(True|False)", explanation, re.IGNORECASE)
    if match:
        return "True" if match.group(1).lower() == "true" else "False"

    zh_match = re.search(r"(预测|结论|最终判断|判定)\s*[:：]?\s*(True|False|属实|真实|真|虚假|不实|假)", explanation,
                         re.IGNORECASE)
    if zh_match:
        tag = zh_match.group(2).lower()
        if tag in ["true", "属实", "真实", "真"]:
            return "True"
        if tag in ["false", "虚假", "不实", "假"]:
            return "False"

    text_all = f"{final_label}\n{explanation}".lower()
    positive_keywords = ["prediction: true", " true", "属实", "真实", "判定为真", "true"]
    negative_keywords = ["prediction: false", " false", "虚假", "不实", "判定为假", "false"]

    pos_idx = max((text_all.rfind(k) for k in positive_keywords), default=-1)
    neg_idx = max((text_all.rfind(k) for k in negative_keywords), default=-1)

    if pos_idx > neg_idx and pos_idx != -1:
        return "True"
    if neg_idx > pos_idx and neg_idx != -1:
        return "False"

    return "Unknown"


def calculate_metrics(y_true, y_pred):
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
    if not valid_pairs:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "total": 0,
            "correct": 0,
            "unknown_count": len(y_true),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    tp = sum(1 for t, p in valid_pairs if t == "True" and p == "True")
    fp = sum(1 for t, p in valid_pairs if t == "False" and p == "True")
    fn = sum(1 for t, p in valid_pairs if t == "True" and p == "False")
    tn = sum(1 for t, p in valid_pairs if t == "False" and p == "False")
    correct = sum(1 for t, p in valid_pairs if t == p)
    total = len(valid_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = correct / total if total else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct,
        "unknown_count": len(y_true) - len(valid_pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_experiment_for_prompt(system, data, prompt_name):
    logger.info("Running prompt: %s", prompt_name)
    system.set_fusion_prompt(prompt_name)

    y_true = []
    y_pred = []
    details = []
    raw_outputs = []

    def process_item(item_with_idx):
        idx, item = item_with_idx
        gt = item["label"]
        claim = item["text"]

        report = None
        pred = "Unknown"
        error_message = None

        for attempt in range(UNKNOWN_RETRY_TIMES + 1):
            try:
                report = system.run_single_inference(claim, source="FinFact")
                pred = extract_prediction(report)
                if pred != "Unknown":
                    break
                error_message = f"Unknown prediction at attempt {attempt + 1}"
            except Exception as e:
                report = {"final_label": "Error", "explanation_path": str(e)}
                pred = "Unknown"
                error_message = f"attempt {attempt + 1} failed: {e}"

            if attempt < UNKNOWN_RETRY_TIMES:
                time.sleep(UNKNOWN_RETRY_DELAY_SECONDS * (attempt + 1))

        if pred == "Unknown":
            pred = UNKNOWN_FALLBACK_LABEL
            fallback_note = f" | fallback->{UNKNOWN_FALLBACK_LABEL}"
            error_message = (error_message or "Unknown prediction") + fallback_note

        return idx, item, gt, claim, report, pred, error_message

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ITEMS) as executor:
        futures = {
            executor.submit(process_item, (idx, item)): (idx, item)
            for idx, item in enumerate(data, start=1)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Prompt={prompt_name}", leave=False):
            idx, item, gt, claim, report, pred, error_message = future.result()

            y_true.append(gt)
            y_pred.append(pred)

            # 提取 pipeline 过程信息
            pipeline_trace = report.get("_pipeline_trace", {}) if isinstance(report, dict) else {}
            retrieved_evidence = pipeline_trace.get("retrieved_evidence", [])
            plan = pipeline_trace.get("plan", {})

            # 格式化检索证据为可读字符串
            retrieved_evidence_text = ""
            if isinstance(retrieved_evidence, list):
                for i, doc in enumerate(retrieved_evidence, 1):
                    content = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
                    score = doc.get("score", "N/A") if isinstance(doc, dict) else "N/A"
                    retrieved_evidence_text += f"[Doc {i} | score={score}] {content}\n"
            else:
                retrieved_evidence_text = str(retrieved_evidence)

            details.append(
                {
                    "prompt_name": prompt_name,
                    "index": idx,
                    "claim": claim,
                    "ground_truth": gt,
                    "prediction": pred,
                    "correct": gt == pred,
                    "raw_label": item.get("raw_label", ""),
                    "error": error_message,
                    # === Pipeline 过程记录 ===
                    "retrieved_evidence": retrieved_evidence_text.strip(),
                    "tool_plan": json.dumps(plan, ensure_ascii=False) if plan else "",
                    "tool_outputs": json.dumps(report.get("tool_outputs_raw", {}), ensure_ascii=False)[:3000] if isinstance(report, dict) else "",
                    "compressed_evidence": report.get("compressed_evidence", "") if isinstance(report, dict) else "",
                    "fusion_system_prompt": report.get("fusion_system_prompt", "") if isinstance(report, dict) else "",
                    "fusion_user_prompt": report.get("fusion_user_prompt", "") if isinstance(report, dict) else "",
                    "fusion_response": report.get("explanation_path", "") if isinstance(report, dict) else "",
                }
            )

            raw_outputs.append(
                {
                    "prompt_name": prompt_name,
                    "index": idx,
                    "claim": claim,
                    "ground_truth": gt,
                    "prediction": pred,
                    "raw_output": report,
                }
            )

    metrics = calculate_metrics(y_true, y_pred)
    logger.info(
        "Prompt=%s | Acc=%.2f%% Precision=%.2f%% Recall=%.2f%% F1=%.2f%%",
        prompt_name,
        metrics["accuracy"] * 100,
        metrics["precision"] * 100,
        metrics["recall"] * 100,
        metrics["f1"] * 100,
    )

    return {
        "prompt_name": prompt_name,
        "metrics": metrics,
        "details": details,
        "raw_outputs": raw_outputs,
    }


def save_results(all_results, sample_size, model_name, prompts_used):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    summary_rows = []
    detail_rows = []
    all_raw = []

    for result in all_results:
        m = result["metrics"]
        summary_rows.append(
            {
                "prompt_name": result["prompt_name"],
                "accuracy": f"{m['accuracy']:.4f}",
                "precision": f"{m['precision']:.4f}",
                "recall": f"{m['recall']:.4f}",
                "f1": f"{m['f1']:.4f}",
                "total_samples": m["total"],
                "correct": m["correct"],
                "unknown_count": m["unknown_count"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
        )
        detail_rows.extend(result["details"])
        all_raw.extend(result["raw_outputs"])

    summary_df = pd.DataFrame(summary_rows)

    summary_file = os.path.join(run_dir, f"summary_{timestamp}.csv")
    details_file = os.path.join(run_dir, f"details_{timestamp}.json")
    full_file = os.path.join(run_dir, f"full_{timestamp}.json")
    raw_file = os.path.join(run_dir, f"raw_outputs_{timestamp}.json")

    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    # details 保存为 JSON（包含大段文本，CSV 无法良好展示）
    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(detail_rows, f, ensure_ascii=False, indent=2)

    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)

    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": model_name,
                "sample_size": sample_size,
                "prompts": prompts_used,
                "results": [
                    {
                        "prompt_name": item["prompt_name"],
                        "metrics": item["metrics"],
                    }
                    for item in all_results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return run_dir, summary_file, details_file, full_file, raw_file


def print_final_table(all_results):
    sorted_results = sorted(all_results, key=lambda x: x["metrics"]["f1"], reverse=True)
    print("\n" + "=" * 90)
    print("Prompt 比较结果")
    print("=" * 90)
    print(f"{'Prompt名称':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 90)
    for item in sorted_results:
        m = item["metrics"]
        print(
            f"{item['prompt_name']:<28} {m['accuracy']:>10.2%} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%}"
        )

    best = sorted_results[0]
    print(f"\n🏆 最佳 Prompt: {best['prompt_name']} (F1: {best['metrics']['f1']:.2%})")


def get_active_prompts():
    if RUN_ALL_PROMPTS:
        return list(PROMPT_TEMPLATES.keys())
    return DEFAULT_PROMPTS


def main():
    active_prompts = get_active_prompts()

    print("=" * 70)
    print("Edition_1 FinFact 多Prompt实验")
    print("=" * 70)
    print(f"模型: {DEFAULT_MODEL}")
    print(f"Prompt数量: {len(active_prompts)}")
    print(f"并发数: {MAX_CONCURRENT_ITEMS}")
    print(f"FinFact测试量: {TEST_PARAMS['FinFact']}")

    data = load_finfact_data(TEST_PARAMS["FinFact"])
    print(f"有效测试样本: {len(data)}")
    print(f"标签分布: {dict(Counter([item['label'] for item in data]))}")

    system = PromptExperimentSystem(model=DEFAULT_MODEL)

    all_results = []
    for prompt_name in active_prompts:
        result = run_experiment_for_prompt(system, data, prompt_name)
        all_results.append(result)

    print_final_table(all_results)

    run_dir, summary_file, details_file, full_file, raw_file = save_results(
        all_results=all_results,
        sample_size=TEST_PARAMS["FinFact"],
        model_name=DEFAULT_MODEL,
        prompts_used=active_prompts,
    )

    print("\n结果已保存：")
    print(f"运行目录: {run_dir}")
    print(f"summary: {summary_file}")
    print(f"details: {details_file}")
    print(f"full: {full_file}")
    print(f"raw_outputs: {raw_file}")


if __name__ == "__main__":
    main()
