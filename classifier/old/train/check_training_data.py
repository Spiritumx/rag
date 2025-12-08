#!/usr/bin/env python3
"""检查训练数据的格式，找出为什么模型不输出标签"""
from datasets import load_from_disk
import json
import re

DATA_DIR = "/root/autodl-tmp/data"

print("=" * 70)
print("检查训练数据格式")
print("=" * 70)

dataset = load_from_disk(DATA_DIR)
train_data = dataset['train']

print(f"\n训练集样本数: {len(train_data)}\n")

# 查看前3个样本的完整格式
for i in range(3):
    sample = train_data[i]
    print(f"\n{'=' * 70}")
    print(f"样本 {i+1}:")
    print('=' * 70)
    print(json.dumps(sample['messages'], indent=2, ensure_ascii=False))

# 分析 assistant 回复的格式
print("\n" + "=" * 70)
print("分析 assistant 回复的结构")
print("=" * 70)

label_positions = []
content_lengths = []

for i, sample in enumerate(train_data[:20]):
    for msg in sample['messages']:
        if msg['role'] == 'assistant':
            content = msg['content']
            content_lengths.append(len(content))

            # 查找标签
            match = re.search(r'\[\[(\w+)\]\]', content)
            if match:
                label = match.group(1)
                label_pos = match.start()
                label_positions.append(label_pos)

                print(f"\n样本 {i+1}:")
                print(f"  内容长度: {len(content)} 字符")
                print(f"  标签: {label}")
                print(f"  标签位置: {label_pos} (占比 {label_pos/len(content)*100:.1f}%)")

                # 显示标签前后的内容
                start = max(0, label_pos - 100)
                end = min(len(content), label_pos + 50)
                print(f"  标签附近文本:")
                print(f"    ...{content[start:end]}...")

                # 显示内容的开头和结尾
                print(f"  内容开头（前150字符）:")
                print(f"    {content[:150]}...")
                print(f"  内容结尾（后100字符）:")
                print(f"    ...{content[-100:]}")
            break

# 统计
if label_positions:
    avg_pos = sum(label_positions) / len(label_positions)
    avg_len = sum(content_lengths) / len(content_lengths)
    print("\n" + "=" * 70)
    print("统计分析:")
    print("=" * 70)
    print(f"平均内容长度: {avg_len:.0f} 字符")
    print(f"平均标签位置: {avg_pos:.0f} 字符 ({avg_pos/avg_len*100:.1f}%)")
    print(f"标签位置范围: {min(label_positions)} - {max(label_positions)} 字符")

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
print("""
如果标签在内容的末尾（>90%），而评估时 max_new_tokens=512 不够长，
那么模型输出会在生成标签之前就被截断，导致 UNKNOWN。

解决方案：
1. 简化训练数据，让 assistant 只输出标签，不要长文本分析
2. 或者增加评估时的 max_new_tokens
3. 或者修改 prompt，让模型先输出标签，再输出分析
""")
