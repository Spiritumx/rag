import os
import re
from collections import defaultdict
import pandas as pd

# 日志文件路径 (根据你上一步的输出)
LOG_FILE = "/root/autodl-tmp/output/eval_final_lora.txt"

def parse_log_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到日志文件 {file_path}")
        return None

    print(f"正在分析日志: {file_path} ...\n")
    
    # 存储细粒度混淆矩阵: matrix[GT][Pred] = Count
    matrix = defaultdict(lambda: defaultdict(int))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 使用正则提取 "GT -> Pred: Count"
    # 格式示例: S1 -> S2: 5
    pattern = re.compile(r'([A-Z0-9]+)\s*->\s*([A-Z0-9]+):\s*(\d+)')
    matches = pattern.findall(content)
    
    all_labels = set()
    for gt, pred, count in matches:
        count = int(count)
        matrix[gt][pred] = count
        all_labels.add(gt)
        all_labels.add(pred)
        
    return matrix, sorted(list(all_labels))

def calculate_metrics(matrix, labels):
    """计算每个类别的 Precision, Recall (Accuracy), F1"""
    stats = []
    
    # 总体统计
    total_samples = 0
    total_correct = 0
    
    for label in labels:
        # True Positive: GT是该类，且预测也是该类
        tp = matrix[label][label]
        
        # False Negative: GT是该类，但预测成了别的 (漏报) -> 影响 Recall
        fn = sum(matrix[label].values()) - tp
        
        # False Positive: GT是别的，但预测成了该类 (误报) -> 影响 Precision
        fp = sum(matrix[gt][label] for gt in matrix) - tp
        
        total = tp + fn
        total_samples += total
        total_correct += tp
        
        # 计算指标
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # 该类别的分类准确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 预测准确度
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        stats.append({
            "Label": label,
            "Sample Count": total,
            "Correct": tp,
            "Accuracy (Recall)": recall, # 也就是该类别的召回率
            "Precision": precision,
            "F1 Score": f1
        })
        
    return stats, total_correct / total_samples if total_samples > 0 else 0

def to_coarse(label):
    """将细粒度标签转换为粗粒度 (S1->S, M1->M, Z0->Z)"""
    if not label or label == "UNKNOWN": return "UNKNOWN"
    return label[0]

def analyze():
    matrix, fine_labels = parse_log_file(LOG_FILE)
    if not matrix: return

    # ==========================
    # 1. 细粒度分析 (Fine-Grained)
    # ==========================
    print("="*60)
    print("【细粒度详细指标】(Z0, S1, S2, ..., M4)")
    print("="*60)
    
    fine_stats, fine_acc = calculate_metrics(matrix, fine_labels)
    df_fine = pd.DataFrame(fine_stats)
    
    # 格式化输出
    print(df_fine.to_string(index=False, formatters={
        "Accuracy (Recall)": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "F1 Score": "{:.2%}".format
    }))
    print("-" * 60)
    print(f"总体细粒度准确率: {fine_acc:.2%}")
    print("\n")

    # ==========================
    # 2. 粗粒度分析 (Coarse-Grained)
    # ==========================
    print("="*60)
    print("【粗粒度详细指标】(Z vs S vs M)")
    print("="*60)
    
    # 构建粗粒度矩阵
    coarse_matrix = defaultdict(lambda: defaultdict(int))
    coarse_labels = set()
    
    for gt in matrix:
        for pred in matrix[gt]:
            c_gt = to_coarse(gt)
            c_pred = to_coarse(pred)
            coarse_matrix[c_gt][c_pred] += matrix[gt][pred]
            coarse_labels.add(c_gt)
            coarse_labels.add(c_pred)
            
    coarse_stats, coarse_acc = calculate_metrics(coarse_matrix, sorted(list(coarse_labels)))
    df_coarse = pd.DataFrame(coarse_stats)
    
    print(df_coarse.to_string(index=False, formatters={
        "Accuracy (Recall)": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "F1 Score": "{:.2%}".format
    }))
    print("-" * 60)
    print(f"总体粗粒度准确率: {coarse_acc:.2%}")
    
    # ==========================
    # 3. 粗粒度混淆矩阵可视化
    # ==========================
    print("\n[粗粒度混淆矩阵详情]")
    sorted_cl = sorted(list(coarse_labels))
    # 表头
    print(f"{'GT \\ Pred':<10} | " + " | ".join([f"{l:<5}" for l in sorted_cl]))
    print("-" * (15 + 8 * len(sorted_cl)))
    
    for gt in sorted_cl:
        row = [f"{coarse_matrix[gt][l]:<5}" for l in sorted_cl]
        print(f"{gt:<10} | " + " | ".join(row))

if __name__ == "__main__":
    analyze()