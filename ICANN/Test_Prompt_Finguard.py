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
TRUE_DATA_PATH = os.path.join(BASE_DIR, "data/FinGuard/Finance_TRUE_150.csv")
FAKE_DATA_PATH = os.path.join(BASE_DIR, "data/FinGuard/Finance_FAKE_150.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "FinGuard")

SLEEP_INTERVAL = 0.1  # API 调用间隔（秒）
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_finguard.json")  # 断点续跑进度文件

# ============================================================
# 初始化 Qwen 客户端
# ============================================================
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 定义多个不同的 Prompt 模板
# ============================================================
PROMPT_TEMPLATES = {
    # ============================================================
    # 🏆 表现最佳的 Prompts (保留)
    # ============================================================
    
#     "cot_stepwise": {
#         "system": """你是一个采用链式思维(Chain-of-Thought)方法的新闻核查AI。
# 你会按照结构化的步骤进行分析，确保判断的严谨性和可追溯性。""",
#         "user": """请使用链式思维方法，逐步分析以下新闻的真实性。

# 【新闻内容】
# {text}

# 【分析步骤】
# Step 1 - 内容摘要：这篇新闻的核心主张是什么？
# Step 2 - 语言风格：使用的语言是客观中立的还是煽动性的？
# Step 3 - 证据评估：文中提供了哪些支持性证据？这些证据可信吗？
# Step 4 - 逻辑检验：论证过程是否合理？是否存在逻辑跳跃？
# Step 5 - 综合判断：基于以上分析，得出结论。

# 完成分析后，输出最终判断：Prediction: True 或 Prediction: False"""
#     },

#     "binary_classifier_en": {
#         "system": """You are a sophisticated binary classifier specialized in detecting fake news and misinformation.
# Your classification is based on linguistic patterns, factual consistency, and source credibility analysis.
# You have been trained on millions of verified real and fake news articles.""",
#         "user": """Analyze the following news article and classify it as authentic or fabricated.

# [NEWS ARTICLE]
# {text}

# [CLASSIFICATION CRITERIA]
# - Linguistic markers: sensationalism, emotional manipulation, clickbait patterns
# - Factual indicators: verifiable claims, credible sources, logical consistency
# - Structural elements: professional journalism standards, balanced reporting

# Provide your binary classification.
# Output format: Prediction: True (authentic) or Prediction: False (fake/misleading)"""
#     },
    
#     "multi_perspective": {
#         "system": """你是一个多角度分析系统，会从不同视角审视新闻的真实性。
# 你会考虑：记者视角、事实核查员视角、普通读者视角、领域专家视角。""",
#         "user": """请从多个角度分析以下新闻的真实性。

# 【新闻内容】
# {text}

# 【多角度分析框架】
# 📰 记者视角：报道是否遵循新闻写作规范？结构是否专业？
# 🔍 事实核查员视角：核心事实是否可验证？数据是否准确？
# 👤 普通读者视角：内容是否试图激起强烈情绪反应？
# 🎓 领域专家视角：专业内容是否准确？术语使用是否正确？

# 综合以上视角，输出判断：Prediction: True 或 Prediction: False"""
#     },

    # ============================================================
    # 🆕 新增 Prompts (基于最佳实践设计)
    # ============================================================

#     "verification_protocol_en": {
#         "system": """You are an automated fact-checking protocol designed to validate financial news.
# You strictly follow a 3-step verification process: Source Check, Content Analysis, and Contextual Consistency.""",
#         "user": """Execute the verification protocol on the following text.

# [TEXT TO VERIFY]
# {text}

# [PROTOCOL STEPS]
# 1. [Source Check] Does the text cite reputable entities? Are the citations verifiable?
# 2. [Content Analysis] Is the tone objective? Are there specific numbers/dates?
# 3. [Consistency] Does the information contradict basic financial logic or known market behaviors?

# Conclusion based on protocol (Pass/Fail).
# Final Output: Prediction: True OR Prediction: False"""
#     },

#     "logical_fallacy_check": {
#         "system": """你是一位逻辑侦探，专门寻找新闻报道中的逻辑谬误。
# 只要通过严密的逻辑推演，往往能发现虚假新闻的破绽。""",
#         "user": """请分析以下新闻是否存在逻辑谬误，并判断其真实性。

# 【新闻原文】
# {text}

# 【逻辑审查】
# 1. 偷换概念：是否混淆了其核心金融概念？
# 2. 循环论证：是否在用结论本身来证明结论？
# 3. 诉诸恐惧/情感：是否试图用恐吓而非事实来说服读者？
# 4. 错误因果：是否强行建立了不相关的因果联系？

# 如果没有发现明显逻辑谬误且事实清晰，则为真。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "editorial_board_vote": {
#         "system": """你模拟一个拥有三位资深成员的新闻编辑委员会：
# 1. 主编（关注整体可信度和新闻价值）
# 2. 合规官（关注监管合规和风险）
# 3. 数据分析师（关注数据合理性）
# 你们需要投票决定这篇新闻是否可以通过真实性审核。""",
#         "user": """编辑委员会请就位，对以下新闻进行审核投票。

# 【新闻稿】
# {text}

# 【委员会讨论】
# - 主编意见：...
# - 合规官意见：...
# - 数据分析师意见：...

# 【最终投票结果】
# 如果至少两票认为真实，则判定为真。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "weighted_evidence_scorer": {
#         "system": """你是一个基于证据权重的评分系统。你会对新闻的真实性要素进行打分（0-10分），总分低于20分（满分30）将被标记为虚假。""",
#         "user": """请对以下新闻进行打分评估。

# 【新闻文本】
# {text}

# 【评分项】
# A. 来源明确性 (0-10): 0=无来源/匿名，10=权威机构实名引用
# B. 细节具体度 (0-10): 0=模糊笼统，10=时间地点人物数据详尽
# C. 叙述中立性 (0-10): 0=极度煽动/主观，10=完全冷静客观

# 请计算总分。
# Decision Rule: Total Score >= 20 -> True; Total Score < 20 -> False.
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "cross_check_simulator": {
#         "system": """You are a research assistant tasked with simulating a cross-reference check.
# Although you cannot browse the live web, use your internal knowledge base to assess if the event aligns with reality.""",
#         "user": """Assess the plausibility of the following news event by simulating a cross-check against established knowledge.

# [News Item]
# {text}

# [Simulation]
# - Search query simulation: What keywords would verify this?
# - Plausibility Check: Is this type of event theoretically possible and consistent with the entities involved?
# - Red Flags: Are there "too good to be true" or "catastrophic" claims without major corroboration?

# Verdict:
# Prediction: True or Prediction: False"""
#     },

    # ============================================================
    # 🆕 融合优化 Prompts (基于 cot_stepwise, multi_perspective, weighted_evidence_scorer)
    # ============================================================

    # 融合1: CoT + 评分系统（每步打分）
    "cot_with_scoring": {
        "system": """你是一个结合链式思维和量化评分的新闻核查系统。
你会按步骤分析，并对每个步骤进行0-10分评估，最终根据总分判断。""",
        "user": """请逐步分析以下新闻，并对每个维度打分。

【新闻内容】
{text}

【分步分析与评分】
Step 1 - 内容核心 (0-10分): 新闻的核心主张是什么？信息是否清晰明确？
评分: ___

Step 2 - 语言风格 (0-10分): 语言是客观中立(10分)还是煽动夸张(0分)？
评分: ___

Step 3 - 证据质量 (0-10分): 是否引用了可验证的来源和具体数据？
评分: ___

Step 4 - 逻辑严密性 (0-10分): 论证过程是否合理？有无逻辑漏洞？
评分: ___

【决策规则】总分 >= 28 → True; 总分 < 28 → False
输出格式：Prediction: True 或 Prediction: False"""
    },

    # 融合2: 多角色 + 评分系统
    "perspective_scoring_panel": {
        "system": """你是一个多角色评分委员会系统。
四位专家从各自角度对新闻真实性进行0-10分评估，汇总决策。""",
        "user": """请四位专家分别对以下新闻进行评分。

【新闻内容】
{text}

【专家评分】
📰 资深记者 (0-10): 报道是否符合新闻写作规范？结构是否专业？
评分: ___

🔍 事实核查员 (0-10): 核心事实是否可验证？数据是否有据可查？
评分: ___

👤 普通读者 (0-10): 内容是否客观？是否试图激起强烈情绪？(情绪化=低分)
评分: ___

💼 金融专家 (0-10): 金融术语和数据是否准确合理？
评分: ___

【决策规则】总分 >= 28 → True; 总分 < 28 → False
输出格式：Prediction: True 或 Prediction: False"""
    },

    # 融合3: CoT + 多角色（分工执行）
    "cot_role_division": {
        "system": """你是一个采用角色分工的链式分析系统。
不同专家负责不同的分析步骤，最终综合各方意见。""",
        "user": """请各角色依次完成分析步骤。

【新闻内容】
{text}

【角色分工分析】
👤 内容编辑 - Step 1: 提取新闻核心主张，概括关键信息点。
分析: ___

📊 数据分析师 - Step 2: 检查文中数据、数字的合理性和可验证性。
分析: ___

🎭 语言学家 - Step 3: 分析语言风格，识别煽动性或情绪化表达。
分析: ___

🔬 逻辑审计师 - Step 4: 检验论证逻辑，寻找推理漏洞。
分析: ___

【综合判断】
基于四位专家的分析，投票决定（多数原则）：
Prediction: True 或 Prediction: False"""
    },

    # 融合4: 三合一综合流程
    "triple_fusion_analyzer": {
        "system": """你是一个三位一体的新闻核查系统，融合：
1. 链式思维的结构化分析
2. 多角度的全面审视
3. 量化评分的客观决策""",
        "user": """请对以下新闻进行三位一体分析。

【新闻内容】
{text}

═══════════════════════════════════════
【第一阶段：链式分析】
Step 1: 核心主张是什么？
Step 2: 证据支撑如何？
Step 3: 逻辑是否自洽？

═══════════════════════════════════════
【第二阶段：多角度审视】
- 记者视角：专业性如何？
- 核查员视角：可验证性如何？
- 读者视角：是否有误导倾向？

═══════════════════════════════════════
【第三阶段：量化评分】
A. 来源可信度 (0-10): ___
B. 内容准确性 (0-10): ___
C. 表达客观性 (0-10): ___
总分: ___/30

【最终裁决】总分>=20且无重大问题 → True; 否则 → False
输出格式：Prediction: True 或 Prediction: False"""
    },

    # 融合5: 增强版CoT（6步细化分析）
    "enhanced_cot_6step": {
        "system": """你是一个采用增强版链式思维的新闻核查AI。
你会通过6个精细化步骤进行分析，确保判断的全面性。""",
        "user": """请使用6步链式思维方法分析以下新闻。

【新闻内容】
{text}

【6步精细分析】
Step 1 - 信息提取：新闻的5W1H（何人、何事、何时、何地、为何、如何）是否完整？

Step 2 - 来源审查：消息来源是谁？是否为权威可信来源？是否可追溯验证？

Step 3 - 语言分析：是否使用中立客观的语言？有无"惊爆""震惊"等煽动性词汇？

Step 4 - 数据核实：涉及的数字、日期、金额等是否具体且合理？

Step 5 - 逻辑检验：论证过程是否严密？是否存在因果谬误或逻辑跳跃？

Step 6 - 动机评估：发布此新闻可能的目的是什么？是否有明显利益驱动？

【综合判断】
基于6步分析，输出：Prediction: True 或 Prediction: False"""
    },

    # 融合6: 增强版评分（5维度）
    "enhanced_5dim_scorer": {
        "system": """你是一个5维度量化评分系统，用于新闻真实性判断。
每个维度0-10分，总分50分，使用严格阈值确保准确性。""",
        "user": """请对以下新闻进行5维度评分。

【新闻内容】
{text}

【5维度评分】
A. 来源权威性 (0-10): 消息来源是否为权威机构或可信人士？
   0=匿名/不可查，10=官方机构/权威媒体
   评分: ___

B. 细节具体度 (0-10): 时间、地点、人物、数据是否具体明确？
   0=模糊笼统，10=详尽精确
   评分: ___

C. 语言中立性 (0-10): 表达是否客观中立？
   0=极度煽动/情绪化，10=完全冷静客观
   评分: ___

D. 逻辑一致性 (0-10): 内容是否前后一致、逻辑自洽？
   0=自相矛盾，10=逻辑严密
   评分: ___

E. 专业准确性 (0-10): 专业术语和概念使用是否正确？
   0=明显错误，10=专业准确
   评分: ___

【决策规则】总分 >= 35 → True; 总分 < 35 → False
输出格式：Prediction: True 或 Prediction: False"""
    },

    # 融合7: 增强版多角色（5角色投票）
    "enhanced_5role_vote": {
        "system": """你模拟一个由5位专家组成的新闻审核委员会。
每位专家独立投票，多数决定最终结果。""",
        "user": """请5位专家对以下新闻进行审核投票。

【新闻内容】
{text}

【专家审核】
👔 主编：从新闻价值和整体可信度判断
   → 投票: True / False

📊 数据分析师：从数据合理性和可验证性判断
   → 投票: True / False

⚖️ 合规官：从法规合规和风险角度判断
   → 投票: True / False

🎓 领域专家：从专业知识准确性判断
   → 投票: True / False

🔍 事实核查员：从信息可追溯性判断
   → 投票: True / False

【投票统计】
True票数: ___ / 5
False票数: ___ / 5

【最终决定】多数票(>=3)获胜
输出格式：Prediction: True 或 Prediction: False"""
    },

    # 融合8: 双轨并行验证
    "dual_track_news_verifier": {
        "system": """You are a dual-track news verification system.
Track A: Qualitative Chain-of-Thought Analysis
Track B: Quantitative Multi-dimensional Scoring
Both tracks run in parallel, results are synthesized for final judgment.""",
        "user": """Execute dual-track verification on the following news.

[NEWS CONTENT]
{text}

══════════════════════════════════════════
[TRACK A: QUALITATIVE ANALYSIS]
1. Core Claim: What is the main assertion?
2. Evidence Check: Are sources cited and verifiable?
3. Language Tone: Objective or sensational?
4. Logic Flow: Any reasoning gaps?
Track A Verdict: Likely True / Likely False

══════════════════════════════════════════
[TRACK B: QUANTITATIVE SCORING]
B1. Source Credibility (0-10): ___
B2. Factual Specificity (0-10): ___
B3. Narrative Objectivity (0-10): ___
Track B Total: ___/30
Track B Verdict (>=20 = True): ___

══════════════════════════════════════════
[SYNTHESIS]
If both tracks agree → Use that verdict
If tracks disagree → Trust Track B (quantitative)

Final Output: Prediction: True or Prediction: False"""
    },

    # 融合9: 阶段式审判流程
    "staged_tribunal": {
        "system": """你是一个三阶段新闻审判系统：
阶段一：初步筛查（快速识别明显问题）
阶段二：深度分析（多维度详细审查）
阶段三：最终裁决（综合评分决策）""",
        "user": """开始对以下新闻进行三阶段审判。

【新闻内容】
{text}

═══════════════════════════════════════
【阶段一：初步筛查】
□ 是否有明确的信息来源？
□ 是否包含可验证的具体信息？
□ 是否存在明显的煽动性语言？
初筛结果: 通过 / 存疑 / 明显虚假

═══════════════════════════════════════
【阶段二：深度分析】(如初筛非"明显虚假")
- 来源可信度分析：...
- 内容逻辑性分析：...
- 专业准确性分析：...
- 语言客观性分析：...

═══════════════════════════════════════
【阶段三：最终裁决】
综合评分 (0-30):
- 可信度: ___/10
- 准确性: ___/10
- 客观性: ___/10
总分: ___/30

【裁决规则】初筛"明显虚假"→False; 总分>=20→True; 否则→False
输出格式：Prediction: True 或 Prediction: False"""
    },

    # ============================================================
    # 🚀 创新设计 Prompts（3个新思路）
    # ============================================================

    # 创新1: 对比假设法（假设真/假，看哪个更合理）
    "hypothesis_contrast": {
        "system": """你是一个采用对比假设法的新闻核查系统。
你会同时假设新闻为真和为假两种情况，分析哪种假设更符合逻辑和证据。""",
        "user": """请使用对比假设法分析以下新闻。

【新闻内容】
{text}

【假设分析】
🔵 假设A：该新闻为真
- 如果为真，需要哪些条件支撑？
- 这些条件是否在新闻中得到体现？
- 该假设的合理性评分 (0-10): ___

🔴 假设B：该新闻为假/误导
- 如果为假，可能的虚假特征有哪些？
- 这些特征是否在新闻中出现？
- 该假设的合理性评分 (0-10): ___

【对比决策】
假设A得分 > 假设B得分 → Prediction: True
假设A得分 <= 假设B得分 → Prediction: False

输出格式：Prediction: True 或 Prediction: False"""
    },

    # 创新2: 红旗检测器（专门寻找虚假特征）
    "red_flag_detector": {
        "system": """你是一个专门检测虚假新闻红旗信号的系统。
你会系统性地检查新闻中常见的虚假/误导特征，累计红旗数量决定判断。""",
        "user": """请对以下新闻进行红旗检测。

【新闻内容】
{text}

【红旗检测清单】
🚩 1. 来源模糊：消息来源不明或无法验证？ □是 □否
🚩 2. 情绪煽动：使用夸张、恐吓、愤怒等情绪化语言？ □是 □否
🚩 3. 细节缺失：关键的时间、地点、人物信息模糊？ □是 □否
🚩 4. 逻辑跳跃：存在因果谬误或不合理推论？ □是 □否
🚩 5. 利益驱动：明显为特定利益群体背书？ □是 □否
🚩 6. 专业错误：金融术语或概念使用明显错误？ □是 □否
🚩 7. 夸大其词："史上最""绝对""100%"等绝对化表述？ □是 □否
🚩 8. 时效可疑：旧闻翻新或时间线混乱？ □是 □否

【红旗统计】
检测到红旗数量: ___ / 8

【决策规则】红旗 <= 2 → True; 红旗 >= 3 → False
输出格式：Prediction: True 或 Prediction: False"""
    },

    # 创新3: 置信度分层决策
    "confidence_tiered_decision": {
        "system": """You are a confidence-based news verification system.
You assess the news across multiple dimensions and calculate a confidence score.
Different confidence levels lead to different decision thresholds.""",
        "user": """Analyze the following news with confidence-based decision making.

[NEWS CONTENT]
{text}

[DIMENSION ANALYSIS]
D1. Source Quality
    - Assessment: (Strong/Moderate/Weak/None)
    - Confidence: (High/Medium/Low)

D2. Factual Precision
    - Assessment: (Detailed/Partial/Vague)
    - Confidence: (High/Medium/Low)

D3. Language Objectivity
    - Assessment: (Neutral/Slight bias/Heavy bias)
    - Confidence: (High/Medium/Low)

D4. Logical Coherence
    - Assessment: (Sound/Minor issues/Major flaws)
    - Confidence: (High/Medium/Low)

[CONFIDENCE SCORING]
High confidence = 3 points, Medium = 2 points, Low = 1 point
Total Confidence Score: ___/12

[TIERED DECISION]
- If Confidence >= 9 AND majority positive assessments → True
- If Confidence >= 9 AND majority negative assessments → False
- If Confidence >= 6 → Lean toward positive assessments
- If Confidence < 6 → Default to False (insufficient evidence)

Final Output: Prediction: True or Prediction: False"""
    },

    # ============================================================
    # ⏸️ 已停用 Prompts (待归档)
    # ============================================================
    
#     "simple": {
#         "system": "你是一个新闻真实性判断助手。只输出判断结果，不需要解释。",
#         "user": """请判断以下新闻的真实性。

# 新闻内容：
# {text}

# 只需回答 True（真实新闻）或 False（虚假新闻），格式：Prediction: True 或 Prediction: False"""
#     },
    
#     "detailed_criteria": {
#         "system": """你是一位经验丰富的新闻事实核查专家，专门识别虚假信息和误导性内容。
# 你具备以下专业能力：
# 1. 识别煽动性、情绪化的语言模式
# 2. 判断信息来源的可靠性和权威性
# 3. 检测逻辑谬误和事实不一致
# 4. 识别政治偏见和议程驱动的内容""",
#         "user": """请运用你的专业知识，对以下新闻进行全面核查。

# 【待核查新闻】
# {text}

# 【核查维度】
# 1. 语言分析：是否存在夸大、煽动性、情绪化表达？是否使用了"惊人"、"震惊"等吸引眼球的词汇？
# 2. 来源可信度：新闻是否引用了可验证的来源？引用的专家或机构是否真实存在？
# 3. 逻辑一致性：报道内容是否自洽？是否存在前后矛盾？
# 4. 事实核查：涉及的数据、日期、人物等是否准确？
# 5. 偏见检测：是否存在明显的政治立场或意识形态倾向？

# 基于以上维度的综合分析，给出最终判断。
# 输出格式：Prediction: True 或 Prediction: False"""
#     },

#     "financial_expert": {
#         "system": """你是一位拥有20年经验的资深金融分析师和新闻核查专家。
# 你曾在华尔街日报、彭博社等权威金融媒体工作，对金融新闻的真实性判断有敏锐的洞察力。
# 你特别擅长识别：市场操纵性假新闻、投资诈骗宣传、夸大的财务数据、虚假的专家背书。""",
#         "user": """作为资深金融新闻核查专家，请对以下新闻进行专业评估。

# 【待评估新闻】
# {text}

# 【专业核查要点】
# 1. 金融数据准确性：涉及的股价、市值、财务数据是否合理？
# 2. 市场影响分析：该新闻是否有操纵市场情绪的意图？
# 3. 来源权威性：消息来源是否为公认的金融机构或监管部门？
# 4. 专业术语使用：金融术语的使用是否正确？是否存在误导性表述？
# 5. 时效性检查：新闻涉及的时间点和事件是否匹配？

# 请给出你的专业判断：Prediction: True 或 Prediction: False"""
#     },

#     "skeptical_investigator": {
#         "system": """你是一位极度怀疑的调查记者，在揭露虚假新闻方面有着丰富的经验。
# 你的座右铭是："非经验证，皆为可疑"。
# 你会从最严格的标准审视每一条新闻，寻找任何可能的漏洞和不实之处。""",
#         "user": """以调查记者的严格标准，审查以下新闻的真实性。

# 【待审查新闻】
# {text}

# 【调查审查清单】
# □ 事实依据：报道中的每个事实陈述是否都有可验证的来源？
# □ 语言中立性：是否使用了中立客观的报道语言？有无情绪煽动？
# □ 逻辑完整性：论证链条是否完整？是否存在逻辑跳跃或隐藏假设？
# □ 利益关联：报道是否可能服务于特定利益群体的议程？
# □ 时间线验证：事件的时间顺序是否合理？是否与已知事实相符？
# □ 专家引用：引用的专家是否真实存在？其言论是否被正确引用？

# 基于严格审查，给出判断：Prediction: True 或 Prediction: False"""
#     },
    
#     "concise": {
#         "system": "新闻真假二分类器。直接输出分类结果。",
#         "user": """新闻：{text}

# 分类结果：Prediction: True 或 Prediction: False"""
#     },
}


def load_data() -> list[dict]:
    """加载两个CSV数据文件，合并为统一格式"""
    data = []
    
    # 加载真实新闻
    df_true = pd.read_csv(TRUE_DATA_PATH)
    text_col = df_true.columns[0]  # 假设第一列是文本
    true_samples = df_true if NUM_SAMPLES < 0 else df_true.head(NUM_SAMPLES)
    for idx, row in true_samples.iterrows():
        data.append({
            "text": str(row[text_col]),
            "label": "True",
            "source": "Finance_TRUE_50"
        })
    
    # 加载虚假新闻
    df_fake = pd.read_csv(FAKE_DATA_PATH)
    text_col = df_fake.columns[0]
    fake_samples = df_fake if NUM_SAMPLES < 0 else df_fake.head(NUM_SAMPLES)
    for idx, row in fake_samples.iterrows():
        data.append({
            "text": str(row[text_col]),
            "label": "False",
            "source": "Finance_FAKE_50"
        })
    
    return data


def build_instruction(text: str, prompt_template: dict) -> tuple[str, str]:
    """根据模板构建指令"""
    system_prompt = prompt_template["system"]
    user_prompt = prompt_template["user"].format(text=text[:2000])  # 限制长度
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


# ============================================================
# 断点续跑功能
# ============================================================
def load_checkpoint() -> dict:
    """加载断点进度"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"✅ 发现断点文件，将从上次中断处继续...")
            print(f"   已完成的Prompt: {list(checkpoint.get('completed_prompts', {}).keys())}")
            return checkpoint
        except Exception as e:
            print(f"⚠️ 加载断点文件失败: {e}，将重新开始")
    return {"completed_prompts": {}, "current_prompt": None, "current_index": 0}


def save_checkpoint(checkpoint: dict):
    """保存断点进度"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """清除断点文件（实验完成后调用）"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🗑️ 断点文件已清除")


def run_experiment_with_prompt(data: list[dict], prompt_name: str, prompt_template: dict,
                                checkpoint: dict = None) -> dict:
    """使用指定的prompt模板运行实验（支持断点续跑）"""
    print(f"\n{'='*60}")
    print(f"测试 Prompt: {prompt_name}")
    print(f"{'='*60}")
    
    y_true = []
    y_pred = []
    details = []
    
    # 检查是否需要从断点恢复
    start_index = 0
    if checkpoint and checkpoint.get("current_prompt") == prompt_name:
        start_index = checkpoint.get("current_index", 0)
        # 恢复已完成的数据
        if "partial_results" in checkpoint:
            partial = checkpoint["partial_results"]
            y_true = partial.get("y_true", [])
            y_pred = partial.get("y_pred", [])
            details = partial.get("details", [])
        if start_index > 0:
            print(f"  📍 从第 {start_index + 1} 条继续（已完成 {start_index} 条）")
    
    for i, item in enumerate(data):
        # 跳过已处理的样本
        if i < start_index:
            continue
            
        system_prompt, user_prompt = build_instruction(item["text"], prompt_template)
        ground_truth = item["label"]
        
        print(f"  处理第 {i + 1}/{len(data)} 条...", end=" ")
        
        response = call_qwen_api(system_prompt, user_prompt)
        prediction = extract_prediction(response)
        
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        is_correct = "✓" if ground_truth == prediction else "✗"
        print(f"真实: {ground_truth}, 预测: {prediction} {is_correct}")
        
        details.append({
            "index": i,
            "source": item["source"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": ground_truth == prediction,
            "text_preview": item["text"][:100] + "..."
        })
        
        # 每处理一条就保存进度
        if checkpoint is not None:
            checkpoint["current_prompt"] = prompt_name
            checkpoint["current_index"] = i + 1
            checkpoint["partial_results"] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "details": details
            }
            save_checkpoint(checkpoint)
        
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
                "source": detail["source"],
                "ground_truth": detail["ground_truth"],
                "prediction": detail["prediction"],
                "correct": detail["correct"],
                "text_preview": detail["text_preview"]
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
    
    print(f"{'Prompt名称':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    
    for result in sorted_results:
        m = result["metrics"]
        print(f"{result['prompt_name']:<20} {m['accuracy']:>10.2%} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%}")
    
    # 找出最佳prompt
    best = sorted_results[0]
    print(f"\n🏆 最佳 Prompt: {best['prompt_name']} (F1: {best['metrics']['f1']:.2%})")


def main():
    print("=" * 60)
    print("FinGuard 数据集 - 多Prompt性能测试实验")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"测试Prompt数量: {len(PROMPT_TEMPLATES)}")
    print()
    
    # 加载断点进度
    checkpoint = load_checkpoint()
    
    # 加载数据
    print("[1/3] 加载数据...")
    data = load_data()
    print(f"      总数据量: {len(data)} 条")
    
    label_dist = Counter([item["label"] for item in data])
    print(f"      标签分布: {dict(label_dist)}")
    
    # 运行所有prompt实验
    print("\n[2/3] 开始多Prompt测试...")
    all_results = []
    
    # 恢复已完成的prompt结果
    completed_prompts = checkpoint.get("completed_prompts", {})
    
    for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
        # 如果该prompt已完成，直接使用缓存结果
        if prompt_name in completed_prompts:
            print(f"\n{'='*60}")
            print(f"⏭️ 跳过已完成的 Prompt: {prompt_name}")
            print(f"{'='*60}")
            all_results.append(completed_prompts[prompt_name])
            continue
        
        # 运行实验
        result = run_experiment_with_prompt(data, prompt_name, prompt_template, checkpoint)
        all_results.append(result)
        
        # 保存该prompt的完整结果到checkpoint
        checkpoint["completed_prompts"][prompt_name] = result
        checkpoint["current_prompt"] = None  # 清除当前进度
        checkpoint["current_index"] = 0
        checkpoint.pop("partial_results", None)
        save_checkpoint(checkpoint)
        print(f"  💾 Prompt [{prompt_name}] 结果已保存到断点文件")
    
    # 保存结果
    print("\n[3/3] 保存结果...")
    save_results(all_results, OUTPUT_DIR)
    
    # 打印最终比较
    print_final_comparison(all_results)
    
    # 清除断点文件
    clear_checkpoint()
    
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
