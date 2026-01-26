#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fin-Fact-FinFact 数据集分析脚本

功能：
1. 统计基本信息（记录数、标签分布、来源分布）
2. 分析文本长度分布（claim、justification、evidence）
3. 提取高频词汇（中英文分词）
4. 分析 evidence 数量分布
5. 统计图片覆盖率
6. 生成数据样本展示

使用方法：
    python analyze_finfact.py

输出：
    - 控制台打印统计信息
    - 保存分析报告到 analysis_report.txt
"""

import json
import os
from collections import Counter
from urllib.parse import urlparse
import re


def load_dataset(file_path='finfact.json'):
    """加载数据集"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return None
    
    print(f"📂 正在加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 成功加载 {len(data)} 条记录\n")
    return data


def basic_statistics(data):
    """基础统计信息"""
    print("=" * 60)
    print("📊 基础统计信息")
    print("=" * 60)
    
    total = len(data)
    print(f"总记录数: {total:,}")
    
    # 标签分布
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    print("\n🏷️  标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        bar = "█" * int(percentage / 2)
        print(f"  {label:8s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # 数据来源
    sources = []
    for item in data:
        url = item.get('url', '')
        domain = urlparse(url).netloc
        if 'snopes' in domain:
            sources.append('Snopes')
        elif 'politifact' in domain:
            sources.append('PolitiFact')
        else:
            sources.append('Other')
    
    source_counts = Counter(sources)
    print("\n🌐 数据来源:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        print(f"  {source:12s}: {count:4d} ({percentage:5.1f}%)")
    
    # 图片统计
    with_images = sum(1 for item in data if item.get('image_data'))
    without_images = total - with_images
    img_percentage = with_images / total * 100
    
    print("\n🖼️  图片覆盖率:")
    print(f"  包含图片:  {with_images:4d} ({img_percentage:5.1f}%)")
    print(f"  无图片:    {without_images:4d} ({100-img_percentage:5.1f}%)")
    
    # 图片数量详细统计
    image_counts = []
    for item in data:
        img_data = item.get('image_data', [])
        if img_data:
            image_counts.append(len(img_data))
    
    if image_counts:
        avg_images = sum(image_counts) / len(image_counts)
        max_images = max(image_counts)
        print(f"  平均图片数: {avg_images:.2f} 张/记录")
        print(f"  最多图片数: {max_images} 张/记录")
    
    print()


def text_length_analysis(data):
    """文本长度分析"""
    print("=" * 60)
    print("📏 文本长度分布分析")
    print("=" * 60)
    
    # Claim 长度
    claim_lengths = []
    claim_char_lengths = []
    for item in data:
        claim = item.get('claim', '')
        words = claim.split()
        claim_lengths.append(len(words))
        claim_char_lengths.append(len(claim))
    
    print("📝 Claim (声明) 统计:")
    print(f"  词数 - 平均: {sum(claim_lengths)/len(claim_lengths):.1f}, "
          f"最小: {min(claim_lengths)}, 最大: {max(claim_lengths)}")
    print(f"  字符数 - 平均: {sum(claim_char_lengths)/len(claim_char_lengths):.0f}, "
          f"最小: {min(claim_char_lengths)}, 最大: {max(claim_char_lengths)}")
    
    # Justification 长度
    just_lengths = []
    just_char_lengths = []
    for item in data:
        just = item.get('justification', '')
        words = just.split()
        just_lengths.append(len(words))
        just_char_lengths.append(len(just))
    
    print("\n📄 Justification (解释) 统计:")
    print(f"  词数 - 平均: {sum(just_lengths)/len(just_lengths):.1f}, "
          f"最小: {min(just_lengths)}, 最大: {max(just_lengths)}")
    print(f"  字符数 - 平均: {sum(just_char_lengths)/len(just_char_lengths):.0f}, "
          f"最小: {min(just_char_lengths)}, 最大: {max(just_char_lengths)}")
    
    # Evidence 长度
    evidence_counts = []
    evidence_total_chars = []
    for item in data:
        evidences = item.get('evidence', [])
        evidence_counts.append(len(evidences))
        
        total_chars = sum(len(e.get('sentence', '')) for e in evidences)
        evidence_total_chars.append(total_chars)
    
    print("\n🔍 Evidence (证据) 统计:")
    print(f"  数量 - 平均: {sum(evidence_counts)/len(evidence_counts):.1f}, "
          f"最小: {min(evidence_counts)}, 最大: {max(evidence_counts)}")
    print(f"  总字符数 - 平均: {sum(evidence_total_chars)/len(evidence_total_chars):.0f}")
    
    # Evidence 数量分布
    evidence_dist = Counter(evidence_counts)
    print("\n  证据数量分布 (Top 10):")
    for count, freq in sorted(evidence_dist.items())[:10]:
        print(f"    {count} 条证据: {freq:3d} 记录")
    
    print()


def topic_analysis(data):
    """主题分析"""
    print("=" * 60)
    print("🏷️  主题 (Issues) 分析")
    print("=" * 60)
    
    # 收集所有主题
    all_issues = []
    for item in data:
        issues = item.get('issues', [])
        all_issues.extend(issues)
    
    total_issues = len(all_issues)
    unique_issues = len(set(all_issues))
    avg_issues = total_issues / len(data)
    
    print(f"主题总数: {total_issues:,}")
    print(f"唯一主题数: {unique_issues}")
    print(f"平均主题数/记录: {avg_issues:.2f}")
    
    # Top 30 主题
    issue_counts = Counter(all_issues)
    print(f"\n📊 Top 30 热门主题:")
    print(f"{'排名':<6} {'主题':<25} {'记录数':<8} {'占比':<8}")
    print("-" * 55)
    
    for rank, (issue, count) in enumerate(issue_counts.most_common(30), 1):
        percentage = count / len(data) * 100
        print(f"{rank:<6} {issue:<25} {count:<8} {percentage:>6.1f}%")
    
    print()


def keyword_analysis(data, top_n=50):
    """高频词分析"""
    print("=" * 60)
    print(f"🔤 高频词汇分析 (Top {top_n})")
    print("=" * 60)
    
    # 提取所有 claim 文本
    all_text = ' '.join(item.get('claim', '') for item in data)
    
    # 简单分词（按空格和标点）
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    
    # 停用词列表（简化版）
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just',
        'don', 'now', 'if', 'said', 'says', 'about', 'up', 'out', 'into'
    }
    
    # 过滤停用词和短词
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    # 统计词频
    word_counts = Counter(filtered_words)
    
    print(f"总词数: {len(words):,}")
    print(f"唯一词数: {len(set(words)):,}")
    print(f"过滤后词数: {len(filtered_words):,}")
    
    print(f"\n📈 Top {top_n} 高频词:")
    print(f"{'排名':<6} {'词汇':<20} {'出现次数':<10}")
    print("-" * 40)
    
    for rank, (word, count) in enumerate(word_counts.most_common(top_n), 1):
        print(f"{rank:<6} {word:<20} {count:<10}")
    
    print()


def sample_display(data, num_samples=3):
    """展示数据样本"""
    print("=" * 60)
    print(f"📝 数据样本展示 (随机 {num_samples} 条)")
    print("=" * 60)
    
    # 每个标签选一条
    samples_by_label = {}
    for item in data:
        label = item['label']
        if label not in samples_by_label:
            samples_by_label[label] = item
    
    for label in ['false', 'true', 'NEI']:
        if label not in samples_by_label:
            continue
        
        item = samples_by_label[label]
        print(f"\n🏷️  标签: {label.upper()}")
        print("-" * 60)
        
        # Claim
        claim = item.get('claim', '')
        print(f"📢 声明 (Claim):")
        print(f"   {claim[:200]}{'...' if len(claim) > 200 else ''}")
        
        # Sci Digest
        digest = item.get('sci_digest', '')
        print(f"\n🔬 科学摘要 (Sci Digest):")
        print(f"   {digest[:200]}{'...' if len(digest) > 200 else ''}")
        
        # Evidence
        evidences = item.get('evidence', [])
        print(f"\n🔍 证据数量: {len(evidences)}")
        if evidences:
            print(f"   第1条: {evidences[0].get('sentence', '')[:150]}...")
        
        # Issues
        issues = item.get('issues', [])
        print(f"\n🏷️  主题标签: {', '.join(issues[:5])}")
        
        # Image
        has_image = bool(item.get('image_data'))
        print(f"\n🖼️  包含图片: {'是' if has_image else '否'}")
        
        # URL
        url = item.get('url', '')
        source = 'Snopes' if 'snopes' in url else 'PolitiFact' if 'politifact' in url else 'Unknown'
        print(f"🌐 来源: {source}")
        print(f"🔗 链接: {url[:60]}...")
    
    print("\n" + "=" * 60)


def save_report(data, output_file='analysis_report.txt'):
    """保存分析报告到文件"""
    import sys
    from io import StringIO
    
    # 重定向输出
    old_stdout = sys.stdout
    sys.stdout = report_output = StringIO()
    
    # 运行所有分析
    print("Fin-Fact-FinFact 数据集分析报告")
    print("=" * 60)
    print(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    basic_statistics(data)
    text_length_analysis(data)
    topic_analysis(data)
    keyword_analysis(data, top_n=30)
    sample_display(data, num_samples=3)
    
    # 恢复输出
    sys.stdout = old_stdout
    
    # 保存到文件
    report_content = report_output.getvalue()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n💾 分析报告已保存到: {output_file}")


def main():
    """主函数"""
    print("\n" + "🔍 " * 20)
    print("   Fin-Fact-FinFact 数据集分析工具")
    print("🔍 " * 20 + "\n")
    
    # 加载数据
    data = load_dataset('finfact.json')
    if data is None:
        return
    
    # 执行分析
    try:
        basic_statistics(data)
        text_length_analysis(data)
        topic_analysis(data)
        keyword_analysis(data, top_n=50)
        sample_display(data, num_samples=3)
        
        # 保存报告
        save_report(data)
        
        print("\n✅ 分析完成！")
        print("📊 查看 analysis_report.txt 获取完整报告")
        
    except Exception as e:
        print(f"\n❌ 分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
