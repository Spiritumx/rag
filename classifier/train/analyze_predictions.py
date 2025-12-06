#!/usr/bin/env python3
"""分析评估日志，找出 UNKNOWN 的原因"""
import re

LOG_FILE = "/root/autodl-tmp/output/evaluation_log_v3.txt"

print("=" * 70)
print("分析评估日志 - 查找 UNKNOWN 原因")
print("=" * 70)

with open(LOG_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取 UNKNOWN 案例
unknown_section = re.search(r'=== UNKNOWN 预测案例.*?===', content, re.DOTALL)
if unknown_section:
    lines = unknown_section.group(0).split('\n')
    print("\n前10个 UNKNOWN 案例的模型输出：\n")

    case_num = 0
    for i, line in enumerate(lines):
        if line.startswith('[案例'):
            case_num += 1
            if case_num > 10:
                break
            # 打印案例编号和 GT
            print(f"\n{line}")
            # 打印接下来的几行（模型输出）
            if i+1 < len(lines):
                print(lines[i+1][:500])  # 只显示前500个字符
                print("-" * 70)

# 统计最常见的错误预测
print("\n\n" + "=" * 70)
print("错误预测统计（非 UNKNOWN）")
print("=" * 70)

error_section = re.search(r'=== 混淆矩阵 \(所有错误方向\) ===(.+?)=== UNKNOWN', content, re.DOTALL)
if error_section:
    error_lines = error_section.group(1).strip().split('\n')
    print("\n前15个最常见的错误：\n")
    for line in error_lines[:15]:
        if line.strip():
            print(line)

# 分析：找出模型输出的共同模式
print("\n\n" + "=" * 70)
print("分析建议")
print("=" * 70)
print("""
根据评估结果，可能的问题：

1. **Token ID 映射错误**：
   训练时多个标签映射到相同的 token ID
   {'S1': 50, 'S2': 50, 'S3': 50, 'S4': 50, 'M1': 44, 'M2': 44, 'M4': 44}
   这导致模型无法区分这些类别

2. **模型输出格式不符合预期**：
   可能模型输出了长文本解释，但没有包含 [[标签]] 格式

3. **训练不充分**：
   可能需要更多 epochs 或调整学习率

**建议的修复方案**：
1. 修复训练脚本中的类别权重计算逻辑
2. 简化模型输出格式（只输出标签，不要长文本）
3. 增加训练轮数或调整超参数
""")
