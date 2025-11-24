#!/bin/bash
set -e

echo "正在创建 raw_data 目录并链接 subsampled 数据..."

datasets=("hotpotqa" "2wikimultihopqa" "musique" "nq" "trivia" "squad")

mkdir -p raw_data

for dataset in "${datasets[@]}"; do
    echo "处理 $dataset ..."
    mkdir -p "raw_data/$dataset"
    
    if [[ -f "processed_data/$dataset/dev_500_subsampled.jsonl" ]]; then
        # 复制文件而不是软链接，避免某些读取问题
        cp "processed_data/$dataset/dev_500_subsampled.jsonl" "raw_data/$dataset/dev.jsonl"
        echo "  -> 已将 dev_500_subsampled.jsonl 复制为 dev.jsonl"
    else
        echo "  -> 警告: processed_data/$dataset/dev_500_subsampled.jsonl 不存在"
    fi
done

echo "完成！现在 evaluate.py 应该能找到 raw_data/$dataset/dev.jsonl 了。"

