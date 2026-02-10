"""
将 Finance_FAKE.csv 按 7:2:1 的比例分成 train, test, validation 数据集
"""
import csv
import random

# 设置随机种子保证可复现
random.seed(42)

# 读取数据
input_path = 'd:/Programming/Project/FMD/ICANN/data/FinGuard/Finance_TRUE.csv'
output_dir = 'd:/Programming/Project/FMD/ICANN/data/FinGuard/FinGuard_TRUE'

with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# 打乱顺序
random.shuffle(data)

n = len(data)
# 按 7:2:1 计算各集合大小
train_size = int(n * 0.7)
test_size = int(n * 0.2)
val_size = n - train_size - test_size  # 剩余的给validation

train_data = data[:train_size]
test_data = data[train_size:train_size + test_size]
val_data = data[train_size + test_size:]

# 获取字段名
fieldnames = list(data[0].keys())

# 保存数据集
def save_csv(filepath, rows, fields):
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

save_csv(f'{output_dir}/Finance_FAKE_train.csv', train_data, fieldnames)
save_csv(f'{output_dir}/Finance_FAKE_test.csv', test_data, fieldnames)
save_csv(f'{output_dir}/Finance_FAKE_validation.csv', val_data, fieldnames)

# 打印统计信息
print("数据集划分完成！")
print(f"\n原始数据集: {len(data)} 条")
print(f"训练集 (train): {len(train_data)} 条 ({len(train_data)/n*100:.1f}%)")
print(f"测试集 (test): {len(test_data)} 条 ({len(test_data)/n*100:.1f}%)")
print(f"验证集 (validation): {len(val_data)} 条 ({len(val_data)/n*100:.1f}%)")
print(f"\n文件已保存到: {output_dir}")
