# README



## Test_Prompt_Finguard.py

| Prompt名称                 | 复杂度 | 特点                                        |
| -------------------------- | ------ | ------------------------------------------- |
| **simple**                 | 简单   | 基础判断指令                                |
| **detailed_criteria**      | 复杂   | 5个核查维度（语言、来源、逻辑、事实、偏见） |
| **cot_stepwise**           | 复杂   | 5步链式思维分析                             |
| **financial_expert**       | 复杂   | 金融专家角色+5个专业核查要点                |
| **binary_classifier_en**   | 中等   | 英文专业分类器                              |
| **skeptical_investigator** | 复杂   | 调查记者角色+6项审查清单                    |
| **concise**                | 最简   | 极简模式                                    |
| **multi_perspective**      | 复杂   | 4种视角多角度分析                           |

**True False各3条**



Prompt名称               Accuracy  Precision     Recall         F1

simple                   83.33%     75.00%    100.00%     85.71%
binary_classifier_en     83.33%    100.00%     66.67%     80.00%
cot_stepwise             66.67%     66.67%     66.67%     66.67%
concise                  66.67%     66.67%     66.67%     66.67%
detailed_criteria        66.67%    100.00%     33.33%     50.00%
financial_expert         66.67%    100.00%     33.33%     50.00%
skeptical_investigator    100.00%      0.00%      0.00%      0.00%
multi_perspective        33.33%      0.00%      0.00%      0.00%

最佳 Prompt: simple (F1: 85.71%)





性能汇总已保存至: FinGuard\prompt_comparison_summary_20260126_182518.csv
详细结果已保存至: FinGuard\prompt_comparison_details_20260126_182518.csv
完整JSON已保存至: FinGuard\prompt_comparison_full_20260126_182518.json

---



**True False各50条**





性能汇总已保存至: FinGuard\prompt_comparison_summary_20260126_230214.csv
详细结果已保存至: FinGuard\prompt_comparison_details_20260126_230214.csv
完整JSON已保存至: FinGuard\prompt_comparison_full_20260126_230214.json

##### 【最终性能比较】
##### Prompt名称               Accuracy  Precision     Recall         F1

**binary_classifier_en**     86.00%     87.50%     84.00%     85.71%
**multi_perspective**        85.51%     87.88%     82.86%     85.29%
**cot_stepwise**             84.69%     84.31%     86.00%     85.15%

simple                   82%	  80.77%	84%	82.35%

concise                  80.00%     78.85%     82.00%     80.39%

detailed_criteria	78.00%    91.18%   62.00%   73.81%

financial_expert         75.00%     93.10%     54.00%     68.35%
skeptical_investigator     76.32%     80.95%     54.84%     65.38%

🏆 最佳 Prompt: binary_classifier_en (F1: 85.71%)





## Test_Prompt_Finfact.py

| 名称                     | 特点            |
| ------------------------ | --------------- |
| `simple`                 | 简洁直接        |
| `detailed_criteria`      | 5维度详细核查   |
| `cot_stepwise`           | 链式思维5步分析 |
| `financial_expert`       | 金融专家视角    |
| `binary_classifier_en`   | 英文分类器      |
| `skeptical_investigator` | 怀疑论调查员    |
| `concise`                | 极简模式        |
| `multi_perspective`      | 多角度分析      |

| 字段                                                         | 说明                         | 使用方式                   |
| ------------------------------------------------------------ | ---------------------------- | -------------------------- |
| [claim](vscode-file://vscode-app/d:/Programming/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) | 待验证的声明                 | 直接使用                   |
| [sci_digest](vscode-file://vscode-app/d:/Programming/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) | 声明的核心摘要（列表）       | 合并为字符串，限制500字符  |
| [justification](vscode-file://vscode-app/d:/Programming/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) | 事实核查的完整论证文本       | 限制2500字符               |
| [evidence](vscode-file://vscode-app/d:/Programming/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) | 支撑论证的可溯源依据（列表） | 格式化为带来源链接的字符串 |



**2True2False**

性能汇总已保存至: FinFact\prompt_comparison_summary_20260126_210819.csv
详细结果已保存至: FinFact\prompt_comparison_details_20260126_210819.csv
完整JSON已保存至: FinFact\prompt_comparison_full_20260126_210819.json


##### 【最终性能比较】
##### Prompt名称                    Accuracy  Precision     Recall         F1
cot_stepwise                 100.00%    100.00%    100.00%    100.00%
detailed_criteria             75.00%    100.00%     50.00%     66.67%
binary_classifier_en          75.00%    100.00%     50.00%     66.67%
concise                       75.00%    100.00%     50.00%     66.67%
simple                        50.00%     50.00%     50.00%     50.00%
financial_expert              50.00%     50.00%     50.00%     50.00%
skeptical_investigator        50.00%     50.00%     50.00%     50.00%
multi_perspective             50.00%     50.00%     50.00%     50.00%

🏆 最佳 Prompt: cot_stepwise (F1: 100.00%)



---



**25True25False**



性能汇总已保存至: FinFact\prompt_comparison_summary_20260126_230744.csv
详细结果已保存至: FinFact\prompt_comparison_details_20260126_230744.csv
完整JSON已保存至: FinFact\prompt_comparison_full_20260126_230744.json



##### 【最终性能比较】
##### Prompt名称                    Accuracy  Precision     Recall         F1
**cot_stepwise**                  66.67%     62.50%     71.43%     66.67%
**multi_perspective**             63.83%     60.71%     73.91%     66.67%
detailed_criteria             62.00%     61.54%     64.00%     62.75%
**binary_classifier_en**          66.00%     70.00%     56.00%     62.22%
concise                       62.00%     62.50%     60.00%     61.22%
simple                        60.00%     60.00%     60.00%     60.00%
financial_expert              54.00%     53.57%     60.00%     56.60%
skeptical_investigator        43.48%     40.00%     36.36%     38.10%

🏆 最佳 Prompt: cot_stepwise (F1: 66.67%)



---

### Finguard: 300条

现在只使用 **cot_stepwise**     **multi_perspective**          **binary_classifier_en**          以及变形,进行测试



共:

- `cot_stepwise` (链式思维)
- `binary_classifier_en` (二分类器-英文)
- `multi_perspective` (多视角分析)
- **`verification_protocol_en`** (英文): 模仿 `binary_classifier_en` 的结构化风格，设定了严格的“验证协议”（Verification Protocol），分三步验证来源、内容和逻辑。
- **`logical_fallacy_check`** (中文): 针对性更强的逻辑分析器，专门寻找偷换概念、循环论证等谬误，这通常是假新闻的硬伤。
- **`editorial_board_vote`** (中文): 扩展了 `multi_perspective` (多视角) 的概念，模拟一个“编辑委员会”投票，引入了合规官和数据分析师的角色。
- **`weighted_evidence_scorer`** (中文): 将定性分析转化为定量打分，设定明确的阈值（20/30分），试图通过量化减少模糊判断。
- **`cross_check_simulator`** (英文): 模拟研究助手进行“交叉验证”，虽然不能联网，但迫使模型调用内部知识库进行一致性检查。

​               

**Finguard300条**

##### Prompt名称               Accuracy  Precision     Recall         F1
binary_classifier_en     84.67%     87.14%     81.33%     84.14%
**cot_stepwise**             85.57%     83.65%     89.26%     86.36%

**multi_perspective**	0.8837	0.8739	0.8981	0.8858
verification_protocol_en	0.8127	0.7831	0.8667	0.8228
logical_fallacy_check	0.8133	0.9196	0.6867	0.7863
editorial_board_vote	0.8227	0.805	0.8533	0.8285
**weighted_evidence_scorer**	0.8633	0.8658	0.86	0.8629
cross_check_simulator	0.7692	0.7299	0.8523	0.7864

**优势分析：**

1. **cot_stepwise**: 5步结构化链式推理（摘要→语言→证据→逻辑→综合）
2. **multi_perspective**: 4视角全面审视（记者、核查员、读者、专家）
3. **weighted_evidence_scorer**: 3维度量化评分（来源、细节、中立性）+ 阈值决策



### Finfact: 100条

- 保留了最佳的 3 个 Prompt：`cot_stepwise`, `binary_classifier_en`, `multi_perspective`.
- **新增了 5 个针对 FinFact 任务设计的 Prompt**：
  - **`verification_protocol_en`**: 严格的三步验证协议（证据匹配、逻辑一致性、语境有效性）。
  - **`logical_fallacy_check`**: 逻辑谬误检测，专门分析声明与证据之间的逻辑断裂。
  - **`editorial_board_vote`**: 模拟三人核查委员会（首席核查员、领域专家、逻辑分析师）进行投票。
  - **`weighted_evidence_scorer`**: 证据权重打分系统，量化证据的覆盖度和一致性。
  - **`cross_check_simulator`**: 模拟交叉验证过程，将提供的 Jusitification/Evidence 视为真理库进行比对。



##### Prompt名称                    Accuracy  Precision     Recall         F1
**verification_protocol_en**      73.33%     77.78%     67.74%     72.41% 
cot_stepwise                  69.77%     63.83%     76.92%     69.77%
multi_perspective             67.71%     64.81%     74.47%     69.31%
binary_classifier_en          65.00%     67.44%     58.00%     62.37%

**weighted_evidence_scorer**      76.00%     78.26%     72.00%     75.00%
**editorial_board_vote**          74.00%     75.00%     72.00%     73.47%
**cross_check_simulator**         73.00%     73.47%     72.00%     72.73%
logical_fallacy_check         59.00%     64.52%     40.00%     49.38%



---



**Finfact 300条**

##### Prompt名称                    Accuracy  Precision     Recall         F1
verification_protocol_en      69.90%     74.79%     59.73%     66.42%		

**editorial_board_vote**	0.7333	0.7465	0.7067	0.726
**weighted_evidence_scorer**	0.7458	0.7626	0.7114	0.7361
**cross_check_simulator**	0.77	0.7755	0.76	0.7677

**三个最佳prompt的核心优势：**

1. **editorial_board_vote**: 多角色投票机制，关注不同维度
2. **weighted_evidence_scorer**: 量化评分系统，明确决策阈值
3. **cross_check_simulator**: 交叉验证框架，系统化检查流程



### 总结：

**Finfact top3**：cross_check_simulator	weighted_evidence_scorer	editorial_board_vote	（300条）



**Finguard top3：** multi_perspective	weighted_evidence_scorer	cot_stepwise        （300条）     





---

### 继续优化prompt

#### Finfact300

| Prompt名称                   | 融合策略          | 核心特点                       |
| ---------------------------- | ----------------- | ------------------------------ |
| `expert_panel_scoring`       | 投票 + 评分       | 三位专家分别评分，总分决定结果 |
| `triple_validation_vote`     | 交叉验证 + 投票   | 三个独立验证器投票，多数决定   |
| `detailed_scoring_validator` | 评分 + 交叉验证   | 四维度精细评分（阈值28/40）    |
| `committee_cross_reference`  | 多角色 + 交叉验证 | 委员会成员分别交叉验证后投票   |
| `enhanced_weighted_scorer`   | 增强版评分        | 五维度评分（阈值35/50）        |
| `systematic_cross_validator` | 增强版交叉验证    | 5步验证协议（≥4步通过）        |
| `tribunal_judgment`          | 三合一审判        | 三阶段流程：审查→评分→投票     |
| `dual_track_verifier`        | 双轨并行          | 量化评分+定性验证矩阵决策      |
| editorial_board_vote         |                   | 多角色投票机制，关注不同维度   |
| weighted_evidence_scorer     |                   | 量化评分系统，明确决策阈值     |
| cross_check_simulator        |                   | 交叉验证框架，系统化检查流程   |



性能汇总已保存至: D:\Programming\Project\FMD\ICANN\FinFact\prompt_comparison_summary_20260130_034215.csv
详细结果已保存至: D:\Programming\Project\FMD\ICANN\FinFact\prompt_comparison_details_20260130_034215.csv
完整JSON已保存至: D:\Programming\Project\FMD\ICANN\FinFact\prompt_comparison_full_20260130_034215.json


##### 【最终性能比较】
##### Prompt名称                    Accuracy  Precision     Recall         F1
**dual_track_verifier**           74.00%     72.22%     78.00%     75.00%
systematic_cross_validator     72.45%     71.15%     75.51%     73.27%
tribunal_judgment            73.00%     72.55%     74.00%     73.27%
detailed_scoring_validator     71.00%     69.81%     74.00%     71.84%
enhanced_weighted_scorer      71.72%     71.43%     71.43%     71.43%
expert_panel_scoring          70.00%     71.74%     66.00%     68.75%
committee_cross_reference     70.00%     73.81%     62.00%     67.39%
triple_validation_vote        68.00%     73.68%     56.00%     63.64%
editorial_board_vote	0.7333	0.7465	0.7067	0.726
**weighted_evidence_scorer**	0.7458	0.7626	0.7114	0.7361
**cross_check_simulator**	0.77	0.7755	0.76	0.7677



```
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
```



```
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
输出格式：Prediction: True 或 Prediction: False"""
    },
```



```
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
Prediction: True or Prediction: False"""
    },
```





#### 继续优化Finguard，使用300





性能汇总已保存至: D:\Programming\Project\FMD\ICANN\FinGuard\prompt_comparison_summary_20260130_221240.csv
详细结果已保存至: D:\Programming\Project\FMD\ICANN\FinGuard\prompt_comparison_details_20260130_221240.csv
完整JSON已保存至: D:\Programming\Project\FMD\ICANN\FinGuard\prompt_comparison_full_20260130_221240.json


##### 【FinGuard300性能比较】
##### Prompt名称               Accuracy  Precision     Recall         F1
**enhanced_cot_6step**       86.86%     84.47%     97.75%     90.62%
**confidence_tiered_decision**     89.00%     88.24%     90.00%     89.11%
**cot_with_scoring**         88.00%     87.50%     88.67%     88.08%
triple_fusion_analyzer     87.02%     84.71%     91.10%     87.79%
**staged_tribunal**          87.25%     87.84%     86.67%     87.25%
enhanced_5dim_scorer     86.67%     83.95%     90.67%     87.18%
perspective_scoring_panel     87.00%     86.75%     87.33%     87.04%
dual_track_news_verifier     86.33%     83.44%     90.67%     86.90%
cot_role_division        81.19%     80.30%     89.83%     84.80%
enhanced_5role_vote      83.33%     80.12%     88.67%     84.18%
hypothesis_contrast      79.26%     72.54%     93.96%     81.87%
red_flag_detector        82.83%     89.34%     74.15%     81.04%

🏆 最佳 Prompt: enhanced_cot_6step (F1: 90.62%)

| Prompt 名称                  | 核心特点                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| `cot_with_scoring`           | 结合链式思维（CoT）和逐步量化评分，对每个分析步骤打分并以总分阈值决定真假。 |
| `perspective_scoring_panel`  | 多角色（记者、核查员、读者、金融专家）独立评分并汇总，总分决定最终判断。 |
| `cot_role_division`          | 将链式分析分配给不同角色（内容、数据、语言、逻辑），由各角色结论投票汇总。 |
| `triple_fusion_analyzer`     | 三阶段融合：结构化CoT分析 + 多角度审视 + 量化评分，三者联合给出结论。 |
| `enhanced_cot_6step`         | 增强版CoT，6步精细化检查（含5W1H、来源、语言、数据、逻辑、动机）以提升覆盖面。 |
| `enhanced_5dim_scorer`       | 五维度量化评分（来源、细节、中立性、逻辑、专业）并用严格阈值判定真伪。 |
| `enhanced_5role_vote`        | 五位专家独立投票（主编、数据、合规、领域、核查），多数票决定结果。 |
| `dual_track_news_verifier`   | 双轨并行：定性CoT分析与定量多维评分同时运行，再按规则合成最终判断。 |
| `staged_tribunal`            | 阶段式审判流程：初筛→深度多维分析→综合评分裁决，短路明显虚假样本。 |
| `hypothesis_contrast`        | 对比假设法：分别假设“真/假”，评估哪一假设更符合证据与逻辑以做决定。 |
| `red_flag_detector`          | 专注红旗信号检测（8项常见虚假特征），通过红旗计数阈值快速判断。 |
| `confidence_tiered_decision` | 基于多维度置信度分层决策：计算置信分并按置信等级应用不同判定阈值。 |



```
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
```

```
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
```

```
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
```

```
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
```

---









# Edition 1内容



## 尝试提升Module的表现

| prompt_name                | accuracy | precision | recall | f1     | correct | tp   | fp   | fn   | tn   |
| -------------------------- | -------- | --------- | ------ | ------ | ------- | ---- | ---- | ---- | ---- |
| dual_track_verifier        | 0.51     | 0.3636    | 0.087  | 0.1404 | 51      | 4    | 7    | 42   | 47   |
| weighted_evidence_scorer   | 0.48     | 0         | 0      | 0      | 48      | 0    | 6    | 46   | 48   |
| cross_check_simulator      | 0.49     | 0.1429    | 0.0217 | 0.0377 | 49      | 1    | 6    | 45   | 48   |
| tribunal_judgment          | 0.44     | 0.4074    | 0.4783 | 0.44   | 44      | 22   | 32   | 24   | 22   |
| editorial_board_vote       | 0.53     | 0.4286    | 0.0652 | 0.1132 | 53      | 3    | 4    | 43   | 50   |
| systematic_cross_validator | 0.54     | 0         | 0      | 0      | 54      | 0    | 0    | 46   | 54   |
| detailed_scoring_validator | 0.54     | 0         | 0      | 0      | 54      | 0    | 0    | 46   | 54   |
| enhanced_weighted_scorer   | 0.54     | 0         | 0      | 0      | 54      | 0    | 0    | 46   | 54   |
| expert_panel_scoring       | 0.54     | 0         | 0      | 0      | 54      | 0    | 0    | 46   | 54   |
| committee_cross_reference  | 0.54     | 0         | 0      | 0      | 54      | 0    | 0    | 46   | 54   |



还是太烂，修改

---

## 修复内容（2026-03-06）

**根本原因：Edition_1 测试链路向 LLM 传入的信息与 ICANN 版本完全不同**

| 问题 | 位置 | 说明 |
|------|------|------|
| `justification` 被垃圾内容替换 | `Fusion_Engine.py` | `_apply_golden_prompt` 把 `{justification}` 填成了压缩工具输出（格式如 `1. 0.`），模型无有效信息可用 |
| `sci_digest` 字段缺失 | `Fusion_Engine.py` + `test.py` | 含 `{sci_digest}` 的模板触发 `KeyError`，被异常分支静默返回 `Prediction: False`，导致 `tp=0` |
| 原始字段未加载 | `test.py` `load_finfact_data` | 数据加载只保留了 `claim` 和 `label`，`justification`/`evidence`/`sci_digest` 全被丢弃 |

### 修改了哪些文件

**`Edition_1/Module_New/test.py`**

1. **`load_finfact_data()`**：新增加载 `sci_digest`（限500字符）、`justification`（限2500字符）、`evidence`（格式化为带序号的字符串）并保存到 data 列表
2. **`run_single_inference()`**：新增 `original_fields` 参数，执行完 `scheduler.execute()` 后将原始字段注入黑板（`orig_justification` / `orig_evidence` / `orig_sci_digest`）
3. **`run_experiment_for_prompt()` 中的 `process_item()`**：调用 `run_single_inference` 时透传 `original_fields`

**`Edition_1/Module_New/Fusion_Engine.py`**

4. **`_apply_golden_prompt()`（FinFact 分支）**：
   - 从黑板读取 `orig_justification` / `orig_evidence` / `orig_sci_digest`
   - 以原始 `justification` 为主要内容，DAG 工具输出作为补充附加（不再替代）
   - 模板 `format()` 新增 `sci_digest=` 参数传入
   - 用 `try/except KeyError` 兜底，不含 `{sci_digest}` 的模板也能正常运行

---



### 修改后finfact10



##### Prompt名称                       Accuracy  Precision     Recall         F1
weighted_evidence_scorer         90.00%     66.67%    100.00%     80.00%
cross_check_simulator            90.00%     66.67%    100.00%     80.00%
editorial_board_vote             90.00%     66.67%    100.00%     80.00%
committee_cross_reference        90.00%     66.67%    100.00%     80.00%
dual_track_verifier              80.00%     50.00%    100.00%     66.67%
systematic_cross_validator       80.00%     50.00%    100.00%     66.67%
detailed_scoring_validator       80.00%     50.00%    100.00%     66.67%
enhanced_weighted_scorer         70.00%     40.00%    100.00%     57.14%
expert_panel_scoring             70.00%     40.00%    100.00%     57.14%
tribunal_judgment                60.00%     33.33%    100.00%     50.00%

🏆 最佳 Prompt: weighted_evidence_scorer (F1: 80.00%)

结果已保存：
运行目录: D:\Programming\Project\FMD\Edition_1\results\20260306_150146
summary: D:\Programming\Project\FMD\Edition_1\results\20260306_150146\summary_20260306_150146.csv
details: D:\Programming\Project\FMD\Edition_1\results\20260306_150146\details_20260306_150146.json
full: D:\Programming\Project\FMD\Edition_1\results\20260306_150146\full_20260306_150146.json
raw_outputs: D:\Programming\Project\FMD\Edition_1\results\20260306_150146\raw_outputs_20260306_150146.json

---

## 进一步优化（2026-03-06，10条测试结果分析）

**测试结果**: 最佳 prompt（cross_check_simulator 等）达到 Accuracy=90%, F1=80%，全部 Recall=100%，但存在 1 个 False Positive

**失败案例分析后发现的新问题**

| 问题                              | 原因                                                         | 影响           |
| --------------------------------- | ------------------------------------------------------------ | -------------- |
| Prompt 做"话题匹配"而非"判决提取" | `cross_check_simulator` 等问"claim 是否存在于知识库"，但 justification 里提到 False 声明是为了驳斥它，模型看到"topic 存在"就预测 True | 所有 FP 的根因 |
| DAG 压缩输出全是 `'0.'` 噪声      | FinFact 的工具链（CGT/SCP 等）是为 FinGuard 新闻设计的，对科学声明核查无实质贡献 | 浪费 API 调用  |

### 修改了哪些文件（第二轮）

**`Edition_1/Module_New/test.py`**

1. **`cross_check_simulator`**：核心问法从"claim 是否存在于知识库"改为"知识库支持还是驳斥该 claim"，新增 Critical rule：若 Ground Truth 标注为 hoax/scam/false/misleading 则直接输出 Prediction: False
2. **`systematic_cross_validator`**：Step 1 从"topic 是否存在"改为"提取事实核查员的明确判决（VERIFIED/DEBUNKED）"
3. **`committee_cross_reference`**：三位委员的分析角度从"事实核查"改为"判决提取"，明确要求先判断 justification 是支持还是驳斥 claim

**`Edition_1/Module_New/Fusion_Engine.py`**

4. **DAG 补充过滤**：检测 `compressed_insights` 是否为纯 `0/1` 格式的无意义输出，若是则跳过追加，不再污染 justification

---







## ICANN和Edition1两个脚本的核心区别

### Test_Prompt_Finfact.py — **基准测试**

```
原始数据集 (claim + justification + evidence)
        ↓
    直接喂给 LLM
        ↓
    Prediction
```

**没有任何中间处理**，就是把数据集里的字段原封不动塞进 prompt，让模型判断。用来测试 prompt 本身的上限。

---

### test.py — **DAG 增强版**

```
原始数据集 (claim + justification + evidence)
        ↓
  ① 用 claim 从 DAG 知识库检索相似条目
  ② Planner 规划要用哪些工具
  ③ Scheduler 执行工具链（CGT/SCP/FCV 等）
  ④ Fusion Engine 汇总 → 喂给 LLM
        ↓
    Prediction
```

**多了三个额外步骤**，理论上能提供更多上下文。

---

## DAG 起了什么作用（目前）

理论意图是：通过多工具并行分析，给 LLM 提供**额外视角**（来源可信度、逻辑问题、专业核实等）。

**实际结果**：对 FinFact **几乎没有贡献**。原因是：

1. **工具是为 FinGuard 设计的**——那些工具（SCP来源可信度、EVA常识验证等）是为"判断新闻真假"设计的，FinFact 是"用已有论证验证学术声明"，根本不需要这些工具
2. **压缩输出是 `"0."` 噪声**——工具给不了有意义的结论，压缩后全是废话
3. **原始数据集的 justification 本身已经是"答案"**——FinFact 数据集里已经有人类专家写好的论证，再检索知识库只会引入不相关内容

---

## 总结

|                 | ICANN              | Edition_1                      |
| --------------- | ------------------ | ------------------------------ |
| 性质            | 纯 Prompt 工程基准 | 尝试用 DAG Agent 增强          |
| 信息来源        | 数据集原始字段     | 数据集字段 + 工具链分析        |
| FinFact 效果    | 更好（~77%）       | 修复前很差，修复后接近基准     |
| FinGuard 适用性 | ✓                  | ✓（工具为此设计）              |
| FinFact 适用性  | ✓                  | 工具链基本无效，靠原始字段兜底 |

**结论**：Edition_1 的 DAG 对 FinGuard（新闻判断）是有意义的，对 FinFact（学术声明+已有论证）意义不大——后者数据集本身已经包含了答案，DAG 的价值在于补充"数据集里没有的信息"。





---

### 修改后的跑finfact100

##### Prompt名称                       Accuracy  Precision     Recall         F1
committee_cross_reference        91.00%     91.11%     89.13%     90.11%
cross_check_simulator            90.00%     92.86%     84.78%     88.64%
dual_track_verifier              86.00%     76.67%    100.00%     86.79%
weighted_evidence_scorer         82.00%     72.58%     97.83%     83.33%
detailed_scoring_validator       78.00%     68.18%     97.83%     80.36%
enhanced_weighted_scorer         76.00%     66.18%     97.83%     78.95%
systematic_cross_validator       80.00%     77.08%     80.43%     78.72%
expert_panel_scoring             74.00%     63.89%    100.00%     77.97%
editorial_board_vote             78.00%     74.00%     80.43%     77.08%
tribunal_judgment                68.00%     59.21%     97.83%     73.77%

🏆 最佳 Prompt: committee_cross_reference (F1: 90.11%)

结果已保存：
运行目录: D:\Programming\Project\FMD\Edition_1\results\20260306_194112
summary: D:\Programming\Project\FMD\Edition_1\results\20260306_194112\summary_20260306_194112.csv
details: D:\Programming\Project\FMD\Edition_1\results\20260306_194112\details_20260306_194112.json
full: D:\Programming\Project\FMD\Edition_1\results\20260306_194112\full_20260306_194112.json
raw_outputs: D:\Programming\Project\FMD\Edition_1\results\20260306_194112\raw_outputs_20260306_194112.json



---



## Final Evaluation 配置（2026-03-06）

基于 val 集 100 条调优结果，切换到 **test 集（574 条）** 进行最终评测，运行 Top-4 Prompt：

| Prompt 名称                 | val-100 F1 |
| --------------------------- | ---------- |
| `committee_cross_reference` | 90.11%     |
| `cross_check_simulator`     | 88.64%     |
| `dual_track_verifier`       | 86.79%     |
| `weighted_evidence_scorer`  | 83.33%     |

**配置变更**：`RUN_ALL_PROMPTS = False`，数据路径改为 `test/finfact_test.json`，`limit = -1`（全量）



这是切换到 **test 集（574 条）**的结果：

##### Prompt名称                       Accuracy  Precision     Recall         F1
cross_check_simulator            86.06%     87.90%     84.30%     86.06%
committee_cross_reference        85.19%     91.60%     78.16%     84.35%
dual_track_verifier              82.58%     78.47%     90.78%     84.18%
weighted_evidence_scorer         79.09%     73.70%     91.81%     81.76%



🏆 最佳 Prompt: cross_check_simulator (F1: 86.06%)

结果已保存：
运行目录: D:\Programming\Project\FMD\Edition_1\results\20260307_010446
summary: D:\Programming\Project\FMD\Edition_1\results\20260307_010446\summary_20260307_010446.csv
details: D:\Programming\Project\FMD\Edition_1\results\20260307_010446\details_20260307_010446.json
full: D:\Programming\Project\FMD\Edition_1\results\20260307_010446\full_20260307_010446.json
raw_outputs: D:\Programming\Project\FMD\Edition_1\results\20260307_010446\raw_outputs_20260307_010446.json
