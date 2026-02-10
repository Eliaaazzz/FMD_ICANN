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

#### Finfact使用100

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

性能汇总已保存至: D:\Programming\Project\FMD\ICANN\FinFact\prompt_comparison_summary_20260130_034215.csv
详细结果已保存至: D:\Programming\Project\FMD\ICANN\FinFact\prompt_comparison_details_20260130_034215.csv
完整JSON已保存至: D:\Programming\Project\FMD\ICANN\FinFact\prompt_comparison_full_20260130_034215.json


##### 【最终性能比较】
##### Prompt名称                    Accuracy  Precision     Recall         F1
**dual_track_verifier**           74.00%     72.22%     78.00%     75.00%
**systematic_cross_validator**     72.45%     71.15%     75.51%     73.27%
**tribunal_judgment**             73.00%     72.55%     74.00%     73.27%
detailed_scoring_validator     71.00%     69.81%     74.00%     71.84%
enhanced_weighted_scorer      71.72%     71.43%     71.43%     71.43%
expert_panel_scoring          70.00%     71.74%     66.00%     68.75%
committee_cross_reference     70.00%     73.81%     62.00%     67.39%
triple_validation_vote        68.00%     73.68%     56.00%     63.64%







#### 继续优化Finguard，使用300





性能汇总已保存至: D:\Programming\Project\FMD\ICANN\FinGuard\prompt_comparison_summary_20260130_221240.csv
详细结果已保存至: D:\Programming\Project\FMD\ICANN\FinGuard\prompt_comparison_details_20260130_221240.csv
完整JSON已保存至: D:\Programming\Project\FMD\ICANN\FinGuard\prompt_comparison_full_20260130_221240.json


##### 【FinGuard最终性能比较】
##### Prompt名称               Accuracy  Precision     Recall         F1
**enhanced_cot_6step**       86.86%     84.47%     97.75%     90.62%
**confidence_tiered_decision**     89.00%     88.24%     90.00%     89.11%
**cot_with_scoring**         88.00%     87.50%     88.67%     88.08%
**triple_fusion_analyzer**     87.02%     84.71%     91.10%     87.79%
staged_tribunal          87.25%     87.84%     86.67%     87.25%
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

