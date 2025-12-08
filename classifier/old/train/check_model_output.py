#!/usr/bin/env python3
"""直接读取并显示评估日志的关键部分"""
import sys

LOG_FILE = "/root/autodl-tmp/output/evaluation_log_v3.txt"

print("=" * 70)
print("查看模型实际输出")
print("=" * 70)

try:
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\n总行数: {len(lines)}")
    print(f"\n前50行内容:\n")
    print("=" * 70)

    for i, line in enumerate(lines[:50], 1):
        print(f"{i:3d}: {line.rstrip()}")

    print("\n" + "=" * 70)
    print("\n搜索 UNKNOWN 案例...")

    in_unknown_section = False
    unknown_count = 0

    for i, line in enumerate(lines):
        if 'UNKNOWN 预测案例' in line:
            in_unknown_section = True
            print(f"\n找到 UNKNOWN 部分，从第 {i+1} 行开始\n")
            print("=" * 70)

        if in_unknown_section:
            print(line.rstrip())
            if line.startswith('[案例'):
                unknown_count += 1
                if unknown_count >= 5:  # 只显示前5个案例
                    break

    if not in_unknown_section:
        print("\n未找到 'UNKNOWN 预测案例' 部分")
        print("\n搜索包含 'GT:' 的行（前20行）:")
        print("=" * 70)
        count = 0
        for line in lines:
            if 'GT:' in line and 'Pred:' in line:
                print(line.rstrip())
                count += 1
                if count >= 20:
                    break

except FileNotFoundError:
    print(f"\n错误: 找不到文件 {LOG_FILE}")
    print("请确保已经运行过评估脚本: python classifier/train/evaluate.py")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
