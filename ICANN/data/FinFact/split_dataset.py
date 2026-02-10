"""
将 finfact.json 按 7:2:1 的比例分成 train, test, validation 数据集
使用分层抽样确保各数据集中不同label的比例相似
"""
import json
import random
from collections import defaultdict

# 设置随机种子保证可复现
random.seed(42)

# 读取数据
with open('finfact.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按label分组
label_groups = defaultdict(list)
for item in data:
    label = item.get('label')
    label_groups[label].append(item)

# 初始化结果集
train_data = []
test_data = []
val_data = []

# 对每个label进行分层抽样
for label, items in label_groups.items():
    # 打乱顺序
    random.shuffle(items)
    
    n = len(items)
    # 按 7:2:1 计算各集合大小
    train_size = int(n * 0.7)
    test_size = int(n * 0.2)
    val_size = n - train_size - test_size  # 剩余的给validation
    
    train_data.extend(items[:train_size])
    test_data.extend(items[train_size:train_size + test_size])
    val_data.extend(items[train_size + test_size:])

# 打乱各数据集内部顺序
random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(val_data)

# 保存数据集
with open('finfact_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('finfact_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

with open('finfact_validation.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

# 打印统计信息
print("数据集划分完成！")
print(f"\n原始数据集: {len(data)} 条")
print(f"训练集 (train): {len(train_data)} 条")
print(f"测试集 (test): {len(test_data)} 条")
print(f"验证集 (validation): {len(val_data)} 条")

# 验证各数据集的label分布
from collections import Counter

print("\n各数据集label分布:")
for name, dataset in [("原始数据", data), ("训练集", train_data), ("测试集", test_data), ("验证集", val_data)]:
    labels = [item['label'] for item in dataset]
    counts = Counter(labels)
    total = len(dataset)
    print(f"\n{name}:")
    for label in ['false', 'true', 'NEI']:
        count = counts.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
