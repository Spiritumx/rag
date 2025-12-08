#!/usr/bin/env python
"""
独立的预测脚本，直接执行predict命令的核心逻辑。
等价于: python runner.py SYSTEM MODEL DATASET predict --prompt_set 1 --sample_size 500 --llm_port_num PORT
"""

import os
import re
import json
import copy
import itertools
import argparse
import subprocess
from pathlib import Path
from typing import Dict

# 设置transformers缓存目录
cache_dir = os.path.dirname(os.getcwd()) + '/cache'
# Use HF_HOME instead of deprecated TRANSFORMERS_CACHE
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir  # Keep for backward compatibility

# ============================================================================
# 常量定义 (从 run.py 提取)
# ============================================================================

INSTANTIATION_SCHEMES = {
    "nor_qa": {},
    "oner": {"bm25_retrieval_count": ["15"]},
    "oner_qa": {
        "bm25_retrieval_count": ["15"],
        "distractor_count": ['"1"'],
        "retrieval_mode": ["bm25", "hnsw", "splade", "hybrid"],
    },
    "ircot": {
        "bm25_retrieval_count": ["6"],      # 修改为单个值，节省token消耗 (原本: ["4", "6", "8"])
        "distractor_count": ['"1"'],        # 修改为单个值，节省token消耗 (原本: ['"1"', '"2"', '"3"'])
        "retrieval_mode": ["bm25", "hnsw", "splade", "hybrid"],
    },
    "ircot_qa": {
        "bm25_retrieval_count": ["6"],
        "distractor_count": ['"1"'],
        "retrieval_mode": ["bm25", "hnsw", "splade", "hybrid"],
    },
}


# ============================================================================
# 核心函数
# ============================================================================

def hash_str(string: str) -> str:
    """生成字符串的哈希值"""
    import hashlib
    return str(int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 10**8)


def get_config_file_path_from_name(experiment_name: str) -> str:
    """根据实验名查找配置文件路径"""
    matching_result = list(Path(".").rglob("**/*" + experiment_name + ".jsonnet"))
    matching_result = [
        _result
        for _result in matching_result
        if os.path.splitext(os.path.basename(_result))[0] == experiment_name
    ]
    matching_result = [i for i in matching_result if 'backup' not in str(i)]
    
    if len(matching_result) != 1:
        raise Exception(f"找不到唯一匹配的配置文件 ({experiment_name})，匹配结果数量: {len(matching_result)}")
    
    return str(matching_result[0])


def infer_source_target_prefix(config_filepath: str, evaluation_path: str) -> str:
    """推断 source-target 前缀"""
    # 简化实现：如果 evaluation_path 和 config 数据集相同则返回空
    return ""


def is_experiment_complete(
    original_experiment_file_path: str,
    prediction_file_path: str,
    metrics_file_path: str,
    variable_replacements: str
) -> bool:
    """检查实验是否完成"""
    if not os.path.exists(original_experiment_file_path):
        return False
    if not os.path.exists(prediction_file_path):
        return False
    if not os.path.exists(metrics_file_path):
        return False
    
    # 检查预测文件的完整性
    try:
        with open(prediction_file_path, "r") as file:
            predictions = json.load(file)
            num_complete_items = sum([bool(value) for key, value in predictions.items()])
        return num_complete_items / len(predictions) > 0.9
    except:
        return False


# ============================================================================
# 主要逻辑
# ============================================================================

def run_predictions(system: str, model: str, dataset: str, sample_size: int, llm_port_num: str, force: bool = False):
    """
    运行所有超参数配置的预测
    
    Args:
        system: 系统类型
        model: 模型类型
        dataset: 数据集名称
        sample_size: 样本大小
        llm_port_num: LLM服务端口号
        force: 是否强制重新预测
    """
    # 1. 构造实验名和路径
    if model == "none":
        experiment_name = f"{system}_{dataset}"
    else:
        experiment_name = f"{system}_{model.replace('-', '_')}_{dataset}"
    
    print(f"\n{'='*60}")
    print(f"实验名称: {experiment_name}")
    print(f"{'='*60}\n")
    
    # 2. 确定评估数据路径
    set_name = f"dev_{sample_size}"
    evaluation_path = os.path.join("processed_data", dataset, f"{set_name}_subsampled.jsonl")
    
    if not os.path.exists(evaluation_path):
        print(f"错误: 评估数据文件不存在: {evaluation_path}")
        return
    
    print(f"评估数据路径: {evaluation_path}\n")
    
    # 3. 查找所有已实例化的配置文件
    hyperparameter_variations_directory = os.path.join("instantiated_configs")
    
    # 获取超参数方案
    instantiation_scheme = INSTANTIATION_SCHEMES[system]
    
    # 查找基础配置名
    try:
        base_config_filepath = get_config_file_path_from_name(experiment_name)
        base_config_name = os.path.splitext(os.path.split(base_config_filepath)[1])[0]
    except Exception as e:
        print(f"错误: {e}")
        return
    
    # 4. 遍历所有超参数组合，执行预测
    prediction_count = 0
    skipped_count = 0
    
    for values in itertools.product(*list(instantiation_scheme.values())):
        # 构建当前超参数的变量替换字典
        variable_replacements = {}
        for key, value in zip(instantiation_scheme.keys(), values):
            variable_replacements[key] = copy.deepcopy(value)
        
        # 构建文件名（与 write 时的逻辑一致）
        local_names = ["prompt_set_1"]
        for key, value in variable_replacements.items():
            value_name = value.replace('"', "")
            local_names.append(f"{key}__{value_name}")
        
        overall_local_name = "___".join(local_names).lstrip("_")
        
        # 生成配置文件路径
        local_file_path = os.path.join(
            hyperparameter_variations_directory,
            "____".join([base_config_name, overall_local_name]) + ".jsonnet"
        )
        
        # 处理文件名过长的情况
        local_file_path_basename = os.path.basename(local_file_path)
        if len(local_file_path_basename) > 255:
            local_file_path = os.path.join(
                hyperparameter_variations_directory,
                os.path.splitext(local_file_path_basename)[0][:237]
                + "__"
                + hash_str(local_file_path_basename)
                + ".jsonnet",
            )
        
        if not os.path.exists(local_file_path):
            print(f"警告: 配置文件不存在，跳过: {local_file_path}")
            continue
        
        # 构建预测输出路径
        experiment_name_variant = os.path.splitext(os.path.split(local_file_path)[1])[0]
        prediction_directory = os.path.join("predictions", experiment_name_variant)
        
        prediction_file_name = os.path.splitext(os.path.basename(evaluation_path))[0]
        prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")
        
        evaluation_file_name = os.path.splitext(os.path.split(evaluation_path)[1])[0]
        metrics_file_path = os.path.join(
            prediction_directory, "evaluation_metrics__" + evaluation_file_name + ".json"
        )
        
        # 检查是否需要跳过（默认行为：--skip_if_exists --silent）
        skip_if_exists = not force
        if skip_if_exists and is_experiment_complete(
            local_file_path, prediction_file_path, metrics_file_path, ""
        ):
            skipped_count += 1
            continue
        
        # 执行预测命令
        run_command = f"python predict.py {local_file_path} {evaluation_path} --set_name {set_name} --llm_port_num {llm_port_num} --silent"
        
        if force:
            run_command += " --force"
        
        print(f"\n>>> 执行预测 ({prediction_count + 1})")
        print(f"配置: {os.path.basename(local_file_path)}")
        print(run_command)
        
        subprocess.call(run_command, shell=True)
        prediction_count += 1
    
    print(f"\n{'='*60}")
    print(f"预测完成统计:")
    print(f"  - 执行预测: {prediction_count} 个配置")
    print(f"  - 已跳过: {skipped_count} 个配置（已存在完整结果）")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="运行预测（等价于 runner.py predict 命令）"
    )
    parser.add_argument(
        "system",
        type=str,
        choices=["ircot", "ircot_qa", "oner", "oner_qa", "nor_qa"],
        help="系统类型"
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["flan-t5-xxl", "flan-t5-xl", "gpt", "none"],
        help="模型类型"
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"],
        help="数据集名称"
    )
    parser.add_argument(
        "sample_size",
        type=int,
        help="样本大小（如 500）"
    )
    parser.add_argument(
        "llm_port_num",
        type=str,
        help="LLM 服务端口号",
        default="8010"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新预测，即使结果已存在"
    )
    
    args = parser.parse_args()
    
    # 执行预测
    run_predictions(args.system, args.model, args.dataset, args.sample_size, args.llm_port_num, args.force)


if __name__ == "__main__":
    main()
