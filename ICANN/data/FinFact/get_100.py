import json
import random
from pathlib import Path

# 设置随机种子以保证可重复性
random.seed(42)

# 读取原始数据
data_path = Path("d:/Programming/Project/FMD/ICANN/data/FinFact/finfact.json")
output_path = Path("d:/Programming/Project/FMD/ICANN/data/FinFact/finfact_50.json")

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 筛选有效数据（只保留包含 claim 和 label 为 true/false 的条目）
valid_data = []
for item in raw_data:
    label = item.get("label", "").strip().lower()
    if label in ["true", "false"] and item.get("claim") and item.get("justification"):
        valid_data.append(item)

print(f"有效数据总数: {len(valid_data)}")

# 分别统计 true 和 false 的数量
true_samples = [d for d in valid_data if d["label"].lower() == "true"]
false_samples = [d for d in valid_data if d["label"].lower() == "false"]

print(f"True 样本数: {len(true_samples)}")
print(f"False 样本数: {len(false_samples)}")

# 平衡采样：各取50条
num_each = 25
sampled_true = random.sample(true_samples, min(num_each, len(true_samples)))
sampled_false = random.sample(false_samples, min(num_each, len(false_samples)))

# 合并并打乱
sampled_data = sampled_true + sampled_false
random.shuffle(sampled_data)

print(f"采样后总数: {len(sampled_data)}")
print(f"  - True: {len([d for d in sampled_data if d['label'].lower() == 'true'])}")
print(f"  - False: {len([d for d in sampled_data if d['label'].lower() == 'false'])}")

# 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"\n已保存到: {output_path}")