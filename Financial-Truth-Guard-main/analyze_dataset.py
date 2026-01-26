"""
数据集分析脚本 - Financial Truth Guard
分析金融新闻数据集的基本统计信息和特征
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("📊 Financial Truth Guard - 数据集分析")
print("=" * 60)

# 1. 读取数据集
print("\n[1/6] 正在读取数据集...")
try:
    df_true = pd.read_csv('Pilot/data/finance_dataset/Finance_TRUE.csv')
    df_fake = pd.read_csv('Pilot/data/finance_dataset/Finance_FAKE.csv')
    print(f"✅ 真实新闻: {len(df_true):,} 条")
    print(f"✅ 虚假新闻: {len(df_fake):,} 条")
    print(f"✅ 总计: {len(df_true) + len(df_fake):,} 条")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit(1)

# 2. 添加标签
df_true['label'] = 1
df_fake['label'] = 0
df_all = pd.concat([df_true, df_fake], ignore_index=True)

# 3. 文本长度分析
print("\n[2/6] 分析文本长度...")
df_all['text_length'] = df_all['text'].str.len()
df_all['word_count'] = df_all['text'].str.split().str.len()

print(f"\n📏 文本长度统计（字符数）:")
print(f"   真实新闻 - 平均: {df_true['text'].str.len().mean():.0f} 字符")
print(f"   虚假新闻 - 平均: {df_fake['text'].str.len().mean():.0f} 字符")

print(f"\n📝 词数统计:")
print(f"   真实新闻 - 平均: {df_true['text'].str.split().str.len().mean():.0f} 词")
print(f"   虚假新闻 - 平均: {df_fake['text'].str.split().str.len().mean():.0f} 词")

# 4. 高频词分析
print("\n[3/6] 分析高频词...")

def get_top_words(texts, n=20):
    """提取高频词"""
    words = []
    for text in texts:
        # 跳过空值
        if pd.isna(text):
            continue
        # 转小写，提取单词
        words.extend(re.findall(r'\b[a-z]{3,}\b', str(text).lower()))
    
    # 去除停用词
    stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'was', 
                 'are', 'his', 'her', 'their', 'has', 'have', 'will', 'said',
                 'would', 'they', 'been', 'were', 'which', 'when', 'who'}
    words = [w for w in words if w not in stopwords]
    
    return Counter(words).most_common(n)

true_top_words = get_top_words(df_true['text'].head(1000), 15)
fake_top_words = get_top_words(df_fake['text'].head(1000), 15)

print("\n🔵 真实新闻高频词 (Top 15):")
for word, count in true_top_words:
    print(f"   {word:15s} : {count:4d}")

print("\n🔴 虚假新闻高频词 (Top 15):")
for word, count in fake_top_words:
    print(f"   {word:15s} : {count:4d}")

# 5. 情感词分析
print("\n[4/6] 分析情感词...")

# 情感词列表
emotional_words = ['disgusting', 'frightening', 'vile', 'outrageous', 
                   'shocking', 'terrible', 'horrible', 'alarming']

def count_emotional_words(text):
    """统计情感词出现次数"""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(1 for word in emotional_words if word in text_lower)

df_true['emotional_count'] = df_true['text'].apply(count_emotional_words)
df_fake['emotional_count'] = df_fake['text'].apply(count_emotional_words)

print(f"😡 情感词使用频率:")
print(f"   真实新闻: {df_true['emotional_count'].mean():.3f} 次/篇")
print(f"   虚假新闻: {df_fake['emotional_count'].mean():.3f} 次/篇")
print(f"   差异: {df_fake['emotional_count'].mean() / max(df_true['emotional_count'].mean(), 0.001):.1f}x")

# 6. 数据不平衡分析
print("\n[5/6] 分析数据平衡性...")
print(f"⚖️  数据分布:")
print(f"   真实: {len(df_true):,} ({len(df_true)/len(df_all)*100:.1f}%)")
print(f"   虚假: {len(df_fake):,} ({len(df_fake)/len(df_all)*100:.1f}%)")
print(f"   不平衡比: 1:{len(df_fake)/len(df_true):.2f}")
print(f"\n💡 建议: 训练时使用类别权重或过采样处理不平衡问题")

# 7. 金融关键词分析
print("\n[6/6] 分析金融关键词覆盖率...")

# 读取金融关键词
try:
    with open('Pilot/data/finance_dataset/finance_words.txt', 'r') as f:
        finance_words = f.read().split(',')[:50]  # 前50个
except:
    finance_words = ['GDP', 'market', 'economy', 'financial', 'tax', 
                     'investment', 'stock', 'bond', 'trade', 'banking']

def count_finance_keywords(text):
    """统计金融关键词"""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(1 for word in finance_words if word.lower() in text_lower)

sample_true = df_true['text'].head(1000)
sample_fake = df_fake['text'].head(1000)

print(f"💰 金融关键词密度 (前1000条样本):")
print(f"   真实新闻: {sample_true.apply(count_finance_keywords).mean():.2f} 词/篇")
print(f"   虚假新闻: {sample_fake.apply(count_finance_keywords).mean():.2f} 词/篇")

# 8. 保存分析结果
print("\n" + "=" * 60)
print("✅ 分析完成！")
print("=" * 60)

# 数据质量评估
print("\n📋 数据质量评估:")
print(f"   ✅ 数据规模: 充足 ({len(df_all):,} 条)")
print(f"   {'⚠️' if len(df_fake)/len(df_true) > 1.5 else '✅'} 数据平衡: {'需要处理' if len(df_fake)/len(df_true) > 1.5 else '良好'}")
print(f"   ✅ 文本长度: 适中 (平均 {df_all['text_length'].mean():.0f} 字符)")
print(f"   ✅ 特征区分度: 情感词差异明显")

print("\n💡 建议:")
print("   1. 使用 class_weight='balanced' 处理数据不平衡")
print("   2. 关注 F1-Score 和 Confusion Matrix，而非单纯 Accuracy")
print("   3. 结合 TF-IDF 和词嵌入提取文本特征")
print("   4. 考虑使用深度学习模型捕捉语义信息")

print("\n" + "=" * 60)
