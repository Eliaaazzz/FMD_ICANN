# FMD-test: Qwen3-max 金融信息真假判别实验

## 📁 项目结构

```
FMD-test/
├── .venv/                          # Python 虚拟环境 (uv 创建)
├── Data_FinGuard/                  # 数据集目录
│   ├── Finance_TRUE.csv            # 真实金融新闻数据
│   ├── Finance_FAKE.csv            # 虚假金融新闻数据
│   ├── finance_words.csv
│   └── finance_words.txt
├── qwen_finance_partial_eval.py    # 随机采样评测脚本 (推荐)
├── qwen_finance_eval.py            # 全量评测脚本
└── README.md                       # 本文件
```

## 🔧 环境准备

### 1. 安装 uv (如未安装)

```powershell
# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. 创建虚拟环境

```powershell
cd FMD-test
uv venv .venv
```

### 3. 安装依赖

```powershell
uv pip install openai tqdm --python .venv\Scripts\python.exe
```

## 🚀 运行实验

### 随机采样评测 (推荐，默认各100条)

```powershell
uv run --python .venv\Scripts\python.exe qwen_finance_partial_eval.py --api_key <你的API_KEY>
```

### 自定义采样数量

```powershell
uv run --python .venv\Scripts\python.exe qwen_finance_partial_eval.py --api_key <你的API_KEY> --sample_size 50
```

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_key` | (必需) | Qwen API Key |
| `--model` | `qwen3-max` | 使用的模型名称 |
| `--sample_size` | `100` | 每类采样数量 |
| `--seed` | `42` | 随机种子 (确保可复现) |
| `--true_csv` | `Data_FinGuard/Finance_TRUE.csv` | 真实新闻数据路径 |
| `--fake_csv` | `Data_FinGuard/Finance_FAKE.csv` | 虚假新闻数据路径 |
| `--output_dir` | 自动生成 | 结果输出目录 |

## 📊 输出结果

运行后会在当前目录生成 `qwen_eval_partial_<时间戳>/` 文件夹，包含：

- `metrics.json` - 完整评估指标 (JSON 格式)
- `summary.md` - 评估报告 (Markdown 格式)
- `predictions.csv` - 逐条预测结果
- `errors.txt` - 错误记录

### 评估指标

- **Accuracy**: 整体准确率
- **Precision**: 精确率 (宏平均)
- **Recall**: 召回率 (宏平均)
- **F1**: F1 分数 (宏平均)
- **混淆矩阵**: TP, TN, FP, FN

## 🔑 获取 API Key

1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 创建 API Key
3. Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`

## 📝 复现实验步骤

1. 克隆/下载本项目
2. 确保 `Data_FinGuard/` 目录下有数据文件
3. 按上述步骤创建环境并安装依赖
4. 使用相同的 `--seed 42` 和 `--sample_size 100` 运行
5. 对比 `metrics.json` 中的指标

## ⚠️ 注意事项

- 默认随机种子为 42，确保采样可复现
- API 调用可能产生费用，建议先用小样本测试
- 数据集字段名为 `text`，如有变化请用 `--text_col` 指定
