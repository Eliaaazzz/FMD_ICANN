# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "openai",
# ]
# ///
import json
import re
import time
import pandas as pd
from datetime import datetime
from openai import OpenAI
from collections import Counter
import os

# ============================================================
# 超参数设置
# ============================================================
NUM_SAMPLES = -1  # 【可调】每类测试样本数，设为 -1 表示全部测试
MODEL_NAME = "qwen3-max"
API_KEY = "sk-6234f2144f4946fa81cbfaf6e382c3a0"

# 获取脚本所在目录，确保路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/FinFact/finfact_100.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "FinFact")

SLEEP_INTERVAL = 0.1  # API 调用间隔（秒）

# ============================================================
# 初始化 Qwen 客户端
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 定义多个不同的 Prompt 模板
# 使用字段说明:
# - claim: 待验证的声明
# - sci_digest: 声明的核心摘要
# - justification: 事实核查的完整论证文本
# - evidence: 支撑论证的可溯源依据
# ============================================================
PROMPT_TEMPLATES = {
    # ============================================================
    # 🏆 表现最佳的 Prompts (保留)
    # ============================================================
    
#     "cot_stepwise": {
#         "system": """你是一个采用链式思维(Chain-of-Thought)方法的事实核查AI。
# 你会按照结构化的步骤进行分析，确保判断的严谨性和可追溯性。""",
#         "user": """请使用链式思维方法，逐步分析以下声明的真实性。

# 【声明内容】
# {claim}

# 【声明摘要】
# {sci_digest}

# 【事实核查论证】
# {justification}

# 【支撑证据】
# {evidence}

# 【分析步骤】
# Step 1 - 声明解析：这个声明的核心主张是什么？摘要是否准确概括了关键点？
# Step 2 - 证据评估：提供的证据是否充分？来源是否可靠？
# Step 3 - 论证分析：事实核查论证是否支持或反驳该声明？
# Step 4 - 语境检验：声明是否被正确理解？是否存在断章取义？
# Step 5 - 综合判断：基于以上分析，该声明是否属实？

# 完成分析后，输出最终判断：Prediction: True 或 Prediction: False"""
#     },

#     "binary_classifier_en": {
#         "system": """You are a sophisticated binary classifier specialized in fact-checking claims.
# Your classification is based on evidence analysis, logical consistency, and source credibility.
# You have been trained on millions of verified fact-check cases.""",
#         "user": """Analyze the following claim and classify it as true or false based on the provided evidence.

# [CLAIM]
# {claim}

# [CLAIM SUMMARY]
# {sci_digest}

# [FACT-CHECK ANALYSIS]
# {justification}

# [SUPPORTING EVIDENCE]
# {evidence}

# [CLASSIFICATION CRITERIA]
# - Accuracy: Does the claim accurately represent the facts?
# - Context: Is the claim presented in proper context?
# - Evidence: Does the evidence support or refute the claim?
# - Sources: Are the cited sources credible and verifiable?
# - Misleading: Is the claim potentially misleading?

# Provide your binary classification.
# Output format: Prediction: True (claim is accurate) or Prediction: False (claim is false/misleading)"""
#     },
    
#     "multi_perspective": {
#         "system": """你是一个多角度分析系统，会从不同视角审视声明的真实性。
# 你会考虑：事实核查员视角、领域专家视角、普通读者视角、批判性思维视角。""",
#         "user": """请从多个角度分析以下声明的真实性。

# 【声明内容】
# {claim}

# 【声明摘要】
# {sci_digest}

# 【事实核查论证】
# {justification}

# 【支撑证据】
# {evidence}

# 【多角度分析框架】
# 🔍 事实核查员视角：声明的核心事实是否可验证？证据是否充分？来源是否可靠？
# 🎓 领域专家视角：从专业角度看，声明是否准确？术语使用是否正确？
# 👤 普通读者视角：声明是否可能误导不了解背景的读者？
# 🧠 批判性思维视角：论证逻辑是否严密？是否存在逻辑谬误？证据链是否完整？

# 综合以上视角，输出判断：Prediction: True 或 Prediction: False"""
#     },

    # ============================================================
    # 🆕 新增 Prompts (基于最佳实践设计)
    # ============================================================

#     "verification_protocol_en": {
#         "system": """You are an automated fact-checking protocol.
# You strictly follow a 3-step verification process: Evidence Matching, Logical Consistency, and Contextual Validity.""",
#         "user": """Execute the verification protocol on the following claim.

# [CLAIM TO VERIFY]
# {claim}

# [REFERENCE MATERIALS]
# Summary: {sci_digest}
# Justification: {justification}
# Evidence: {evidence}

# [PROTOCOL STEPS]
# 1. [Evidence Matching] Does concepts and data in the claim strictly align with the provided justification and evidence? (Pass/Fail)
# 2. [Logical Consistency] Is the conclusion logically derived from the premise without fallacies? (Pass/Fail)
# 3. [Contextual Validity] Is the claim presented without omitting crucial context that changes its meaning? (Pass/Fail)

# Conclusion based on protocol.
# Final Output: Prediction: True OR Prediction: False"""
#     },

#     "logical_fallacy_check": {
#         "system": """你是一位逻辑侦探，专门寻找声明中的逻辑谬误。
# 通过对比声明与证据，寻找推理链条中的断裂处。""",
#         "user": """请分析以下声明是否相对于证据存在逻辑谬误。

# 【待验证声明】
# {claim}

# 【核查依据】
# 摘要：{sci_digest}
# 论证：{justification}
# 证据：{evidence}

# 【逻辑审查】
# 1. 过度以此类推：声明是否夸大了证据所支持的范围？
# 2. 证据缺失：声明的关键要素是否有对应的证据支持？
# 3. 断章取义：声明是否忽略了论证中的限制条件或前提？
# 4. 错误归因：是否错误地建立了因果关系？

# 如果没有发现明显逻辑谬误且证据支持，则为真。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "editorial_board_vote": {
#         "system": """你模拟一个拥有三位资深成员的事实核查委员会：
# 1. 首席核查员（关注证据链的完整性）
# 2. 领域专家（关注术语和概念的准确性）
# 3. 逻辑分析师（关注推理过程的严密性）
# 你们需要投票决定该声明是否属实。""",
#         "user": """委员会请就位，对以下声明进行审核投票。

# 【声明】
# {claim}

# 【案卷材料】
# 摘要：{sci_digest}
# 论证：{justification}
# 证据：{evidence}

# 【委员会讨论】
# - 首席核查员意见：...
# - 领域专家意见：...
# - 逻辑分析师意见：...

# 【最终投票结果】
# 如果至少两票认为属实(True)，则判定为真。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "weighted_evidence_scorer": {
#         "system": """你是一个基于证据权重的评分系统。你会对声明的可信度要素进行打分（0-10分），总分低于20分（满分30）将被标记为False。""",
#         "user": """请对以下声明进行基于证据的打分评估。

# 【声明】
# {claim}

# 【证据材料】
# {justification}
# {evidence}

# 【评分项】
# A. 证据覆盖度 (0-10): 0=无直接证据，10=证据完全覆盖声明的所有细节
# B. 一致性 (0-10): 0=声明与证据矛盾，10=声明与证据高度一致
# C. 语境准确性 (0-10): 0=严重断章取义，10=完全忠实于原意

# 请计算总分。
# Decision Rule: Total Score >= 20 -> True; Total Score < 20 -> False.
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "cross_check_simulator": {
#         "system": """You are a research assistant simulating a cross-referencing process.
# You use the provided "Justification" and "Evidence" as your ground truth knowledge base to verify the "Claim".""",
#         "user": """Verify the claim by cross-referencing it against the provided ground truth.

# [Target Claim]
# {claim}

# [Ground Truth Knowledge Base]
# {justification}
# {evidence}

# [Simulation]
# - Initial Check: Does the claim exist in the Knowledge Base?
# - Detail Verification: Do specific numbers, dates, and entities match exactly?
# - Conflict Detection: Is there any statement in the Knowledge Base that directly contradicts the claim?

# Verdict:
# Prediction: True or Prediction: False"""
#     },

    # ============================================================
    # 🆕 融合优化 Prompts (基于最佳表现prompt设计)
    # ============================================================

    # 融合1: 多角色投票 + 评分系统
    "expert_panel_scoring": {
        "system": """你是一个由三位专家组成的评分委员会系统。
每位专家从自己的专业角度对声明进行0-10分评估，最终根据总分决定真实性。
- 证据分析师：评估证据的充分性和可靠性
- 逻辑审计师：评估推理过程的严密性
- 语境专家：评估声明是否被正确理解和呈现""",
        "user": """请三位专家分别对以下声明进行评分。

【声明】
{claim}

【摘要】
{sci_digest}

【论证材料】
{justification}

【证据】
{evidence}

【专家评分】
📊 证据分析师评分 (0-10): 证据是否充分支持或反驳声明？来源是否可靠？
📊 逻辑审计师评分 (0-10): 从证据到结论的推理是否严密？是否存在逻辑谬误？
📊 语境专家评分 (0-10): 声明是否在正确语境下被理解？是否断章取义？

【决策规则】
总分 >= 20 → True（声明属实）
总分 < 20 → False（声明虚假/误导）

输出格式：Prediction: True 或 Prediction: False"""
    },

    # 融合2: 交叉验证 + 投票机制
    "triple_validation_vote": {
        "system": """You are a triple-validation system with three independent validators:
1. Evidence Matcher: Checks if claim aligns with provided evidence
2. Logic Validator: Verifies logical consistency between claim and justification  
3. Context Checker: Ensures claim is not taken out of context
Each validator votes True or False. Majority wins.""",
        "user": """Run triple validation on the following claim.

[CLAIM]
{claim}

[VALIDATION MATERIALS]
Summary: {sci_digest}
Justification: {justification}
Evidence: {evidence}

[VALIDATION PROCESS]
🔍 Evidence Matcher Vote:
- Does the claim's content match the evidence provided?
- Are specific facts, numbers, dates accurate?
→ Vote: True/False

🔍 Logic Validator Vote:
- Is the claim logically supported by the justification?
- Are there any logical fallacies?
→ Vote: True/False

🔍 Context Checker Vote:
- Is the claim presented in proper context?
- Does it accurately represent the source material?
→ Vote: True/False

[FINAL DECISION]
Count votes: If >= 2 True votes → Prediction: True
Otherwise → Prediction: False"""
    },

    # 融合3: 评分系统 + 交叉验证的细节检查
    "detailed_scoring_validator": {
        "system": """你是一个精细化评分验证系统。
你会对声明的每个关键维度进行交叉验证并打分，确保评估的全面性和准确性。""",
        "user": """请对以下声明进行精细化评分验证。

【待验证声明】
{claim}

【验证知识库】
摘要：{sci_digest}
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

【决策阈值】总分 >= 28 → True; 总分 < 28 → False
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
Summary: {sci_digest}
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

    # 优化5: 增强版评分系统（更细致的维度）
    "enhanced_weighted_scorer": {
        "system": """你是一个增强版证据权重评分系统。
你会从5个核心维度对声明进行评分，每个维度0-10分，总分50分。
使用更严格的阈值确保判断的准确性。""",
        "user": """请对以下声明进行五维度评分评估。

【声明】
{claim}

【参考材料】
摘要：{sci_digest}
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
总分 >= 35 → Prediction: True
总分 < 35 → Prediction: False"""
    },

    # 优化6: 增强版交叉验证（更系统化的检查）
    "systematic_cross_validator": {
        "system": """You are an advanced cross-validation system with a systematic 5-step verification protocol.
You treat the justification and evidence as the ground truth and systematically verify every aspect of the claim.""",
        "user": """Execute the systematic cross-validation protocol.

[TARGET CLAIM]
{claim}

[GROUND TRUTH DATABASE]
Summary: {sci_digest}
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

    # 创新7: 审判模式（综合三种方法的结构化流程）
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
摘要：{sci_digest}
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
- 阶段一无重大问题 + 阶段二总分>=20 + 阶段三多数支持True → Prediction: True
- 否则 → Prediction: False"""
    },

    # 创新8: 双轨验证系统（中英双语验证）
    "dual_track_verifier": {
        "system": """You are a dual-track verification system that performs both quantitative scoring and qualitative cross-referencing in parallel, then synthesizes results for final judgment.
Track A: Quantitative Scoring (0-30 scale)
Track B: Qualitative Cross-Reference (Pass/Fail checks)""",
        "user": """Execute dual-track verification on the following claim.

[CLAIM]
{claim}

[REFERENCE MATERIALS]
Summary: {sci_digest}
Justification: {justification}
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
| >= 22         | >= 2           | True           |
| >= 25         | >= 1           | True           |
| < 22          | 3              | True           |
| Otherwise     | -              | False          |

Final Output: Prediction: True or Prediction: False"""
    },

    # ============================================================
    # ⏸️ 已停用 Prompts (待归档)
    # ============================================================

#     "simple": {
#         "system": "你是一个事实核查助手。只输出判断结果，不需要解释。",
#         "user": """请判断以下声明的真实性。

# ## 待验证声明
# {claim}

# ## 声明摘要
# {sci_digest}

# ## 事实核查论证
# {justification}

# 只需回答 True（声明属实）或 False（声明虚假/误导），格式：Prediction: True 或 Prediction: False"""
#     },
    
#     "detailed_criteria": {
#         "system": """你是一位经验丰富的事实核查专家，专门识别虚假信息和误导性内容。
# 你具备以下专业能力：
# 1. 识别煽动性、情绪化的语言模式
# 2. 判断信息来源的可靠性和权威性
# 3. 检测逻辑谬误和事实不一致
# 4. 识别政治偏见和议程驱动的内容""",
#         "user": """请运用你的专业知识，对以下声明进行全面核查。

# 【待核查声明】
# {claim}

# 【声明核心摘要】
# {sci_digest}

# 【事实核查论证】
# {justification}

# 【支撑证据】
# {evidence}

# 【核查维度】
# 1. 声明准确性：声明是否准确反映了事实？是否存在断章取义或曲解？
# 2. 证据充分性：提供的证据是否足以支持或反驳该声明？
# 3. 逻辑一致性：论证过程是否自洽？是否存在逻辑漏洞？
# 4. 来源可信度：证据来源是否可靠？引用是否准确？
# 5. 误导性检测：声明是否具有误导公众的倾向？

# 基于以上维度的综合分析，给出最终判断。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },
    
#     "financial_expert": {
#         "system": """你是一位拥有20年经验的资深金融分析师和事实核查专家。
# 你曾在华尔街日报、彭博社等权威金融媒体工作，对金融相关声明的真实性判断有敏锐的洞察力。
# 你特别擅长识别：市场操纵性假新闻、投资诈骗宣传、夸大的财务数据、政策误读。""",
#         "user": """作为资深金融事实核查专家，请对以下声明进行专业评估。

# 【待评估声明】
# {claim}

# 【声明核心摘要】
# {sci_digest}

# 【详细论证分析】
# {justification}

# 【可溯源证据】
# {evidence}

# 【专业核查要点】
# 1. 事实准确性：声明中涉及的事实、数据是否准确？
# 2. 语境完整性：声明是否在正确的语境下被理解？
# 3. 证据可靠性：引用的证据来源是否权威可信？
# 4. 专业解读：从金融/经济专业角度，该声明是否存在误导？
# 5. 影响评估：该声明是否可能误导公众做出错误判断？

# 请给出你的专业判断：Prediction: True 或 Prediction: False"""
#     },
    
#     "skeptical_investigator": {
#         "system": """你是一位极度怀疑的调查记者，在揭露虚假声明方面有着丰富的经验。
# 你的座右铭是："非经验证，皆为可疑"。
# 你会从最严格的标准审视每一个声明，寻找任何可能的漏洞和不实之处。""",
#         "user": """以调查记者的严格标准，审查以下声明的真实性。

# 【待审查声明】
# {claim}

# 【声明摘要】
# {sci_digest}

# 【核查论证材料】
# {justification}

# 【可溯源证据清单】
# {evidence}

# 【调查审查清单】
# □ 事实核实：声明中的每个事实陈述是否都有证据支持？
# □ 证据溯源：提供的证据来源是否可追溯、可验证？
# □ 引用准确性：原话或原意是否被正确引用？
# □ 语境完整性：声明是否在完整语境下呈现？
# □ 逻辑严密性：从证据到结论的推理是否严密？
# □ 误导可能性：声明是否存在误导公众的风险？

# 基于严格审查，给出判断：Prediction: True 或 Prediction: False"""
#     },
    
#     "concise": {
#         "system": "事实核查二分类器。直接输出分类结果。",
#         "user": """声明：{claim}
# 摘要：{sci_digest}
# 论证：{justification}

# 分类结果：Prediction: True 或 Prediction: False"""
#     },
}



def format_evidence(evidence_list: list[dict]) -> str:
    """格式化证据列表为字符串
    
    每条证据包含:
    - sentence: 证据描述
    - hrefs: 来源链接列表
    """
    if not evidence_list:
        return "无可用证据"
    
    # 确保是列表类型
    if not isinstance(evidence_list, list):
        return "无可用证据"

    formatted = []
    for i, ev in enumerate(evidence_list, 1):
        # 确保ev是字典类型
        if not isinstance(ev, dict):
            continue
            
        sentence = ev.get("sentence", "") or ""
        hrefs = ev.get("hrefs", []) or []
        
        # 过滤掉None值并确保都是字符串
        valid_hrefs = [str(h) for h in hrefs if h is not None]
        
        href_str = ", ".join(valid_hrefs[:3]) if valid_hrefs else "无来源链接"
        if valid_hrefs and len(valid_hrefs) > 3:
            href_str += f" (等{len(valid_hrefs)}个来源)"
        
        if sentence:  # 只添加有内容的证据
            formatted.append(f"  证据{i}: {sentence}\n    来源: {href_str}")

    return "\n".join(formatted) if formatted else "无可用证据"



def load_data() -> list[dict]:
    """加载FinFact JSON数据
    
    使用字段:
    - claim: 待验证的声明
    - sci_digest: 声明的核心摘要（列表）
    - justification: 事实核查的完整论证文本
    - evidence: 支撑论证的可溯源依据（列表）
    - label: 标签 (true/false)
    """
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 尝试修复不完整的JSON（文件可能被截断）
    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("尝试修复截断的JSON...")
        
        # 找到最后一个完整的对象（以 "}," 或 "}" 结尾）
        # 方法1: 尝试找到最后一个完整的 "},\n  {" 或 "}\n]" 模式
        last_complete = content.rfind('},\n  {')
        if last_complete == -1:
            last_complete = content.rfind('},\n{')
        if last_complete == -1:
            last_complete = content.rfind('}\n]')
        
        if last_complete != -1:
            # 截取到最后一个完整对象，并补上 "]"
            fixed_content = content[:last_complete + 1] + "\n]"
            try:
                raw_data = json.loads(fixed_content)
                print(f"修复成功！加载了 {len(raw_data)} 条记录")
            except json.JSONDecodeError:
                # 方法2: 逐行尝试找到有效的JSON结尾
                lines = content.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == '},':
                        # 去掉最后的逗号，加上 "]"
                        test_content = '\n'.join(lines[:i+1])[:-1] + '\n]'
                        try:
                            raw_data = json.loads(test_content)
                            print(f"修复成功！加载了 {len(raw_data)} 条记录")
                            break
                        except:
                            continue
                    elif lines[i].strip() == '}':
                        test_content = '\n'.join(lines[:i+1]) + '\n]'
                        try:
                            raw_data = json.loads(test_content)
                            print(f"修复成功！加载了 {len(raw_data)} 条记录")
                            break
                        except:
                            continue
                else:
                    raise ValueError("无法修复JSON文件，请检查数据文件格式")
        else:
            raise ValueError("无法修复JSON文件，请检查数据文件格式")
    
    data = []
    for item in raw_data:
        label = item.get("label", "").strip().lower()
        # 只处理 true 和 false 标签，跳过其他（如 NEI）
        if label in ["true", "false"]:
            # 处理 sci_digest（列表转字符串）
            sci_digest = item.get("sci_digest", [])
            if isinstance(sci_digest, list):
                sci_digest_text = " ".join(sci_digest)
            else:
                sci_digest_text = str(sci_digest) if sci_digest else ""
            
            # 处理 evidence（格式化为字符串）
            evidence = item.get("evidence", [])
            evidence_text = format_evidence(evidence)
            
            data.append({
                "claim": item.get("claim", ""),
                "sci_digest": sci_digest_text,
                "justification": item.get("justification", ""),
                "evidence": evidence_text,
                "label": "True" if label == "true" else "False",
                "url": item.get("url", ""),
                "author": item.get("author", "")
            })
    
    # 根据 NUM_SAMPLES 限制样本数
    if NUM_SAMPLES > 0:
        true_samples = [d for d in data if d["label"] == "True"][:NUM_SAMPLES]
        false_samples = [d for d in data if d["label"] == "False"][:NUM_SAMPLES]
        data = true_samples + false_samples
    
    return data


def build_instruction(item: dict, prompt_template: dict) -> tuple[str, str]:
    """根据模板构建指令
    
    使用所有可用字段:
    - claim: 待验证的声明
    - sci_digest: 声明的核心摘要
    - justification: 事实核查的完整论证文本
    - evidence: 支撑论证的可溯源依据
    """
    system_prompt = prompt_template["system"]
    user_prompt = prompt_template["user"].format(
        claim=item["claim"],
        sci_digest=item["sci_digest"][:500] if item["sci_digest"] else "无摘要",
        justification=item["justification"][:2500] if item["justification"] else "无论证",
        evidence=item["evidence"][:1000] if item["evidence"] else "无证据"
    )
    return system_prompt, user_prompt


def call_qwen_api(system_prompt: str, user_prompt: str) -> str:
    """调用 Qwen API 获取预测结果"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用出错: {e}")
        return ""


def extract_prediction(response_text: str) -> str:
    """从模型回复中提取预测标签"""
    if not response_text:
        return "Unknown"
    
    # 尝试匹配 "Prediction: True/False" 格式
    match = re.search(r"Prediction:\s*(True|False)", response_text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        return "True" if label == "true" else "False"
    
    # 备选：直接查找 True/False
    response_lower = response_text.lower()
    
    # 查找最后出现的 true 或 false
    true_pos = response_lower.rfind("true")
    false_pos = response_lower.rfind("false")
    
    if true_pos > false_pos:
        return "True"
    elif false_pos > true_pos:
        return "False"
    
    return "Unknown"


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """计算评估指标"""
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
    
    if len(valid_pairs) == 0:
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
            "total": 0, "correct": 0, "unknown_count": len(y_true)
        }
    
    correct = sum(1 for t, p in valid_pairs if t == p)
    total = len(valid_pairs)
    accuracy = correct / total if total > 0 else 0
    
    # 计算二分类指标 (以 True 为正类)
    tp = sum(1 for t, p in valid_pairs if t == "True" and p == "True")
    fp = sum(1 for t, p in valid_pairs if t == "False" and p == "True")
    fn = sum(1 for t, p in valid_pairs if t == "True" and p == "False")
    tn = sum(1 for t, p in valid_pairs if t == "False" and p == "False")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct,
        "unknown_count": len(y_true) - len(valid_pairs),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def run_experiment_with_prompt(data: list[dict], prompt_name: str, prompt_template: dict) -> dict:
    """使用指定的prompt模板运行实验"""
    print(f"\n{'='*60}")
    print(f"测试 Prompt: {prompt_name}")
    print(f"{'='*60}")
    
    y_true = []
    y_pred = []
    details = []
    
    for i, item in enumerate(data):
        system_prompt, user_prompt = build_instruction(item, prompt_template)
        ground_truth = item["label"]
        
        print(f"  处理第 {i + 1}/{len(data)} 条...", end=" ")
        print(f"[{item['claim'][:40]}...]", end=" ")
        
        response = call_qwen_api(system_prompt, user_prompt)
        prediction = extract_prediction(response)
        
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth}, 预测: {prediction} {is_correct}")
        
        details.append({
            "index": i,
            "claim": item["claim"],
            "url": item["url"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": ground_truth == prediction
        })
        
        time.sleep(SLEEP_INTERVAL)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n【{prompt_name}】结果汇总:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1']:.2%}")
    
    return {
        "prompt_name": prompt_name,
        "system_prompt": prompt_template["system"],
        "user_prompt_template": prompt_template["user"],
        "metrics": metrics,
        "details": details,
        "y_true": y_true,
        "y_pred": y_pred
    }


def save_results(all_results: list[dict], output_dir: str):
    """保存结果到CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存性能汇总表
    summary_data = []
    for result in all_results:
        m = result["metrics"]
        summary_data.append({
            "prompt_name": result["prompt_name"],
            "accuracy": f"{m['accuracy']:.4f}",
            "precision": f"{m['precision']:.4f}",
            "recall": f"{m['recall']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "total_samples": m["total"],
            "correct": m["correct"],
            "unknown_count": m["unknown_count"],
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
            "tn": m.get("tn", 0),
            "system_prompt": result["system_prompt"],
            "user_prompt_template": result["user_prompt_template"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"prompt_comparison_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"\n性能汇总已保存至: {summary_file}")
    
    # 2. 保存详细预测结果
    detailed_data = []
    for result in all_results:
        for detail in result["details"]:
            detailed_data.append({
                "prompt_name": result["prompt_name"],
                "index": detail["index"],
                "claim": detail["claim"],
                "url": detail["url"],
                "ground_truth": detail["ground_truth"],
                "prediction": detail["prediction"],
                "correct": detail["correct"]
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = os.path.join(output_dir, f"prompt_comparison_details_{timestamp}.csv")
    detailed_df.to_csv(detailed_file, index=False, encoding="utf-8-sig")
    print(f"详细结果已保存至: {detailed_file}")
    
    # 3. 保存完整JSON（包含所有信息）
    json_file = os.path.join(output_dir, f"prompt_comparison_full_{timestamp}.json")
    json_output = []
    for result in all_results:
        json_output.append({
            "prompt_name": result["prompt_name"],
            "system_prompt": result["system_prompt"],
            "user_prompt_template": result["user_prompt_template"],
            "metrics": result["metrics"]
        })
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"完整JSON已保存至: {json_file}")
    
    return summary_file, detailed_file, json_file


def print_final_comparison(all_results: list[dict]):
    """打印最终比较结果"""
    print("\n" + "=" * 80)
    print("【最终性能比较】")
    print("=" * 80)
    
    # 按F1分数排序
    sorted_results = sorted(all_results, key=lambda x: x["metrics"]["f1"], reverse=True)
    
    print(f"{'Prompt名称':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    
    for result in sorted_results:
        m = result["metrics"]
        print(f"{result['prompt_name']:<25} {m['accuracy']:>10.2%} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%}")
    
    # 找出最佳prompt
    best = sorted_results[0]
    print(f"\n🏆 最佳 Prompt: {best['prompt_name']} (F1: {best['metrics']['f1']:.2%})")


def main():
    print("=" * 60)
    print("FinFact 数据集 - 多Prompt性能测试实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"测试Prompt数量: {len(PROMPT_TEMPLATES)}")
    print()
    
    # 加载数据
    print("[1/3] 加载数据...")
    data = load_data()
    print(f"      总数据量: {len(data)} 条")
    
    label_dist = Counter([item["label"] for item in data])
    print(f"      标签分布: {dict(label_dist)}")
    
    # 运行所有prompt实验
    print("\n[2/3] 开始多Prompt测试...")
    all_results = []
    
    for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
        result = run_experiment_with_prompt(data, prompt_name, prompt_template)
        all_results.append(result)
    
    # 保存结果
    print("\n[3/3] 保存结果...")
    save_results(all_results, OUTPUT_DIR)
    
    # 打印最终比较
    print_final_comparison(all_results)
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
