import pandas as pd
import os

# 设置路径
data_dir = r'd:\Programming\Project\FMD\ICANN\data\FinGuard'
true_file = os.path.join(data_dir, 'Finance_TRUE.csv')
fake_file = os.path.join(data_dir, 'Finance_FAKE.csv')
true_output = os.path.join(data_dir, 'Finance_TRUE_50.csv')
fake_output = os.path.join(data_dir, 'Finance_FAKE_50.csv')

# 读取并随机采样TRUE数据
print("正在处理Finance_TRUE.csv...")
df_true = pd.read_csv(true_file)
print(f"原始TRUE数据总数: {len(df_true)}")
df_true_sample = df_true.sample(n=50, random_state=42)
df_true_sample.to_csv(true_output, index=False)
print(f"已保存50条TRUE数据到: {true_output}")

# 读取并随机采样FAKE数据
print("\n正在处理Finance_FAKE.csv...")
df_fake = pd.read_csv(fake_file)
print(f"原始FAKE数据总数: {len(df_fake)}")
df_fake_sample = df_fake.sample(n=50, random_state=42)
df_fake_sample.to_csv(fake_output, index=False)
print(f"已保存50条FAKE数据到: {fake_output}")

print("\n完成!")
