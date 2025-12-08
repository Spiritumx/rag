import re
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

# --- 配置 ---
# 必须和 evaluate.py 的路径一致
DATA_DIR = "/root/autodl-tmp/data"
LOG_FILE = "/root/autodl-tmp/output/evaluation_log_v2.txt"

def get_coarse_label(fine_label):
    """将细粒度标签映射为 3 大类"""
    if not fine_label or fine_label == "UNKNOWN": return "UNKNOWN"
    if fine_label.startswith("Z"): return "Z_NoRet"
    if fine_label.startswith("S"): return "S_Single"
    if fine_label.startswith("M"): return "M_Multi"
    return "UNKNOWN"

def parse_log_file(log_path):
    """解析 evaluate.py 生成的日志文件"""
    y_true = []
    y_pred = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取日志中的 GT 和 Pred
    # 格式: GT: S4 | Pred: Z0
    matches = re.findall(r'GT:\s*([ZSM]\d+)\s*\|\s*Pred:\s*([ZSM]\d+|UNKNOWN|EMPTY)', content)
    
    for gt, pred in matches:
        y_true.append(gt)
        y_pred.append(pred)
        
    return y_true, y_pred

def analyze():
    print("正在分析日志文件:", LOG_FILE)
    if not os.path.exists(LOG_FILE):
        print("错误: 找不到日志文件，请先运行 evaluate.py")
        return

    y_true, y_pred = parse_log_file(LOG_FILE)
    print(f"提取到 {len(y_true)} 条预测记录")

    # ------------------------------------------------
    # 1. 细粒度分析 (8类)
    # ------------------------------------------------
    print("\n" + "="*40)
    print("【细粒度评估】 (8 Class Accuracy)")
    print("="*40)
    labels_order = sorted(list(set(y_true + y_pred)))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    # ------------------------------------------------
    # 2. 粗粒度分析 (3类 - 论文核心指标)
    # ------------------------------------------------
    y_true_coarse = [get_coarse_label(l) for l in y_true]
    y_pred_coarse = [get_coarse_label(l) for l in y_pred]
    
    print("\n" + "="*40)
    print("【粗粒度评估】 (3 Class Accuracy - Z/S/M)")
    print("这是你论文真正需要的指标！")
    print("="*40)
    print(classification_report(y_true_coarse, y_pred_coarse, digits=4, zero_division=0))

    # ------------------------------------------------
    # 3. 混淆情况速查
    # ------------------------------------------------
    print("\n【主要混淆方向】")
    # 找出错误的预测
    mistakes = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    mistake_counts = pd.Series(mistakes).value_counts().head(5)
    for (true_l, pred_l), count in mistake_counts.items():
        print(f"真实是 [{true_l}] 但预测成了 -> [{pred_l}]: {count} 次")

if __name__ == "__main__":
    analyze()