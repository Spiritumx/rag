import os
import re
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

# 读取数据 (和之前一样)
DATA_FILE = "/root/graduateRAG/classifier/data/training_data/final_finetuning_dataset.jsonl"
# 读取你的预测结果 (为了省时间，你可以把 evaluate.py 改一下把 y_true 和 y_pred 保存下来，或者重新跑一遍下面的逻辑)
# 这里假设你已经有了 y_true 和 y_pred，或者我们重新加载测试集计算

def map_to_coarse(label):
    """将细粒度标签映射为粗粒度标签"""
    if label in ["S-Sparse", "S-Dense", "S-Hybrid"]:
        return "S"
    return label

# ... (复制 evaluate.py 的加载模型和推理部分) ...
# 在计算 metric 之前，加一步映射：

# 假设 y_true 和 y_pred 已经是列表了
y_true_coarse = [map_to_coarse(y) for y in y_true]
y_pred_coarse = [map_to_coarse(y) for y in y_pred]

print("\n" + "="*50)
print("🚀 Coarse-grained Accuracy (Z vs S vs M)")
print("="*50)
print(f"Accuracy: {accuracy_score(y_true_coarse, y_pred_coarse):.2%}")
print("-" * 50)
print(classification_report(y_true_coarse, y_pred_coarse, digits=4))