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



现在只使用 **cot_stepwise**     **multi_perspective**          **binary_classifier_en**          以及变形,进行测试



​               
