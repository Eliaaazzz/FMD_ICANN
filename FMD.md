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

binary_classifier_en     86.00%     87.50%     84.00%     85.71%
multi_perspective        85.51%     87.88%     82.86%     85.29%
cot_stepwise             84.69%     84.31%     86.00%     85.15%

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
cot_stepwise                  66.67%     62.50%     71.43%     66.67%
multi_perspective             63.83%     60.71%     73.91%     66.67%
detailed_criteria             62.00%     61.54%     64.00%     62.75%
binary_classifier_en          66.00%     70.00%     56.00%     62.22%
concise                       62.00%     62.50%     60.00%     61.22%
simple                        60.00%     60.00%     60.00%     60.00%
financial_expert              54.00%     53.57%     60.00%     56.60%
skeptical_investigator        43.48%     40.00%     36.36%     38.10%

🏆 最佳 Prompt: cot_stepwise (F1: 66.67%)



---





# 数据库生成

***ICANN\DAG\DAG_Construct.py***



### 先划分

| 数据集 | 文件名                    | 数量 | false | true  | NEI   |
| ------ | ------------------------- | ---- | ----- | ----- | ----- |
| 训练集 | `finfact_train.json`      | 2357 | 44.3% | 37.8% | 17.9% |
| 测试集 | `finfact_test.json`       | 673  | 44.3% | 37.9% | 17.8% |
| 验证集 | `finfact_validation.json` | 339  | 44.2% | 37.8% | 18.0% |



| 数据集 | 文件名                        | 数量  | 比例  |
| ------ | ----------------------------- | ----- | ----- |
| 训练集 | `Finance_FAKE_train.csv`      | 10868 | 70.0% |
| 测试集 | `Finance_FAKE_test.csv`       | 3105  | 20.0% |
| 验证集 | `Finance_FAKE_validation.csv` | 1554  | 10.0% |
| 训练集 | `Finance_TRUE_train.csv`      | 19644 | 70%   |







1. **统一特征结构**：
   - 你希望对 finfact 和 finguard（fake/true）三类数据，构建统一字段的RAG数据库。
   - finfact 原始字段如 url、claim、author、posted、justification、evidence、label 等都要提取出来。
   - finguard 只有 text（和label），需要用 LLM（API）尽量提取出与 finfact 对应的字段（如作者、时间、事件、权威来源等），但如果 LLM无法高置信度提取，就写 Null，不能乱猜。
2. **特征一致性与区分**：
   - 所有数据都要有完全一致的字段结构（即使某些字段为 Null）。
   - 对 true/false/NEI 等不同label，特征结构也要一致。
   - 某些特征如“负面修辞/夸大”等，false类要提取，true类也要有该字段但内容为 Null。
3. **特征扩展与验证**：
   - 希望数据库包含比原始数据更多的特征（如事件类型、权威来源、修辞模式等），这些特征要能通过 LLM/规则等方式可靠提取。
   - 你希望后续能验证这些新特征是否提升了下游任务效果（如RAG检索、分类等）。
4. **脚本要求**：
   - 新脚本命名为 DAG_Construct.py，结构参考“构建.py”，但要适配上述需求。
   - 代码要有清晰的特征提取逻辑，不能随意猜测，提取不出来就写 Null。





---

### V1：

新特征如下（可在 DAG_Construct.py 中实现，后续可灵活增删）：

1. **entities**：涉及的公司、人物、机构（LLM提取，finfact原生有，finguard需推断）。
2. **events**：金融事件类型（如“收购”、“股价暴跌”等，LLM提取）。
3. **timestamp**：事件时间（finfact原生有，finguard需推断，推断不出写 Null）。
4. **author/source**：作者或权威信息源（finfact原生有，finguard需推断，推断不出写 Null）。
5. **rhetoric_pattern**：修辞/欺诈模式（如“夸大”、“情感操控”，false类需提取，true类为 Null）。
6. **credibility_score**：可信度分数（可根据label赋值，如true=0.9，false=0.1，NEI=0.5）。
7. **evidence_snippets**：证据片段（finfact原生有，finguard可尝试提取或为 Null）。
8. **justification**：推理逻辑（finfact原生有，finguard可尝试提取或为 Null）。
9. **source_url**：原文链接（finfact原生有，finguard可尝试提取或为 Null）。
10. **label**：true/false/NEI。

后续修改建议：

- 若需增加新特征，只需在脚本中添加字段并调整提取逻辑。
- 若需减少特征，删除相关字段即可。
- 若需调整提取方式（如更换LLM、规则），修改对应函数即可。



数据库存到 ICANN\data\DAG，首批仅处理15条（finfact, finguard_true, finguard_false各5条），并自动区分文件名防止覆盖。请稍等。



### V2：

主要加了**断点重跑**，只需要改这两行配置：

- `RESUME = True/False`
- `RUN_TAG = ...`

脚本在 DAG 下会生成两个文件：
- 数据文件：`dag_db_<RUN_TAG>.jsonl`
- 断点文件：`checkpoint_<RUN_TAG>.json`

---

#### 1) 第一次跑（新任务）
**目标**：新建一份数据库，生成新的断点文件。

设置：
```python
RESUME = True
RUN_TAG = None
```

运行后会自动生成一个时间戳，比如：
```
dag_db_20260210_223156.jsonl
checkpoint_20260210_223156.json
```

---

#### 2) 中途断了，想继续（断点续跑）
**目标**：继续写同一份数据库，从断点位置接着跑。

把 `RUN_TAG` 改成你刚才那次的时间戳（或你想固定的名字）：

```python
RESUME = True
RUN_TAG = "20260210_223156"
```

然后重新运行脚本，它会读取 `checkpoint_20260210_223156.json`，从上次中断的位置继续追加写入 `dag_db_20260210_223156.jsonl`。

---

#### 3) 重新开始（不要断点，重跑全部）
**目标**：清空旧输出，重新跑一遍。

设置：
```python
RESUME = False
RUN_TAG = "20260210_223156"
```

再次运行，会自动删除旧的 `dag_db_20260210_223156.jsonl` 和 `checkpoint_20260210_223156.json`，从头开始生成。

---

#### 4) 建议的最稳用法（固定 run_tag）
建议你在实验时手动写一个固定名字，例如：

```python
RUN_TAG = "exp_v1"
```

这样：
- `dag_db_exp_v1.jsonl` 永远对应这一套实验
- 断点也会写在 `checkpoint_exp_v1.json`

---

如果你愿意，我可以再加一行输出提示（比如“当前断点=xxx”）或在运行结束时打印断点文件路径，方便你直接复制 RUN_TAG。







### **后续**——验证新特征提升性能建议：

- **单独写一个评估脚本（如 DAG_Evaluate.py）**，对比有无新特征的检索/分类/问答等下游任务效果。
- 可采用：准确率、召回率、F1、检索排序、问答质量等指标。
- 支持多版本数据库（如带时间戳区分），便于横向对比。





