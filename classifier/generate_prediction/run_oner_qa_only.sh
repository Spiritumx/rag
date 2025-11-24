#!/usr/bin/env bash

# 专门用于重新生成 oner_qa 预测数据的脚本
# 支持并发执行以加快速度

set -euo pipefail

valid_models=("flan-t5-xxl" "flan-t5-xl" "gpt" "none")
datasets=("hotpotqa" "2wikimultihopqa" "musique" "nq" "trivia" "squad")

usage() {
    echo "用法: $0 MODEL LLM_PORT_NUM [MAX_PARALLEL] [--force]"
    echo "  MODEL        : ${valid_models[*]}"
    echo "  LLM_PORT_NUM : LLM服务端口 (例如: 8010)"
    echo "  MAX_PARALLEL : 最大并发数 (默认: 6，推荐2-8之间)"
    echo "  --force      : 强制重新生成已存在的预测文件"
    echo ""
    echo "示例:"
    echo "  $0 gpt 8010           # 使用默认并发数6"
    echo "  $0 gpt 8010 4         # 使用4个并发"
    echo "  $0 gpt 8010 4 --force # 强制重新生成，使用4个并发"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

MODEL="$1"
LLM_PORT="$2"
MAX_PARALLEL="${3:-6}"
FORCE_FLAG=""

# 检查是否有 --force 参数
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE_FLAG="--force"
        break
    fi
done

# 验证模型
found=0
for valid_model in "${valid_models[@]}"; do
    if [[ "$MODEL" == "$valid_model" ]]; then
        found=1
        break
    fi
done

if [[ $found -eq 0 ]]; then
    echo "错误: 无效的模型 '$MODEL'"
    echo "有效的模型: ${valid_models[*]}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_oner_qa"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/main_${TIMESTAMP}.log"

echo "=========================================================="
echo "重新生成 oner_qa 预测数据"
echo "=========================================================="
echo "模型:         $MODEL"
echo "LLM端口:      $LLM_PORT"
echo "最大并发数:   $MAX_PARALLEL"
echo "强制重新生成: ${FORCE_FLAG:-否}"
echo "日志目录:     $LOG_DIR"
echo "=========================================================="
echo "" | tee "$MAIN_LOG"

# 统计变量
TOTAL_TASKS=${#datasets[@]}
COMPLETED_TASKS=0
FAILED_TASKS=0
declare -a FAILED_DATASETS

# 执行单个数据集的任务
run_dataset() {
    local dataset="$1"
    local task_id="$2"
    local log_file="${LOG_DIR}/oner_qa_${dataset}_${TIMESTAMP}.log"
    
    echo "[$task_id/$TOTAL_TASKS] 开始处理: $dataset" | tee -a "$MAIN_LOG"
    
    # 第1步: 写配置
    echo "  -> 写入配置文件..." >> "$log_file"
    if ! python "${SCRIPT_DIR}/runner.py" oner_qa "$MODEL" "$dataset" write \
        --prompt_set 1 --llm_port_num "$LLM_PORT" >> "$log_file" 2>&1; then
        echo "[$task_id/$TOTAL_TASKS] ✗ 失败: $dataset (配置写入失败)" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    # 第2步: 预测
    echo "  -> 运行预测..." >> "$log_file"
    local predict_cmd="python ${SCRIPT_DIR}/runner.py oner_qa $MODEL $dataset predict --prompt_set 1 --sample_size 500 --llm_port_num $LLM_PORT"
    if [[ -n "$FORCE_FLAG" ]]; then
        predict_cmd="$predict_cmd $FORCE_FLAG"
    fi
    
    if ! eval "$predict_cmd" >> "$log_file" 2>&1; then
        echo "[$task_id/$TOTAL_TASKS] ✗ 失败: $dataset (预测失败)" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    # 第3步: 评估
    echo "  -> 运行评估..." >> "$log_file"
    if ! python "${SCRIPT_DIR}/runner.py" oner_qa "$MODEL" "$dataset" evaluate \
        --prompt_set 1 --sample_size 500 --llm_port_num "$LLM_PORT" >> "$log_file" 2>&1; then
        echo "[$task_id/$TOTAL_TASKS] ✗ 失败: $dataset (评估失败)" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    # 第4步: 汇总
    echo "  -> 汇总结果..." >> "$log_file"
    if ! python "${SCRIPT_DIR}/runner.py" oner_qa "$MODEL" "$dataset" summarize \
        --prompt_set 1 --sample_size 500 --llm_port_num "$LLM_PORT" >> "$log_file" 2>&1; then
        echo "[$task_id/$TOTAL_TASKS] ⚠ 警告: $dataset (汇总失败，但预测已完成)" | tee -a "$MAIN_LOG"
    fi
    
    echo "[$task_id/$TOTAL_TASKS] ✓ 完成: $dataset" | tee -a "$MAIN_LOG"
    return 0
}

# 后台进程管理
declare -A PIDS
running_count=0

# 等待空闲槽位
wait_for_slot() {
    while [[ $running_count -ge $MAX_PARALLEL ]]; do
        for pid in "${!PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                local exit_code=$?
                local dataset="${PIDS[$pid]}"
                
                if [[ $exit_code -eq 0 ]]; then
                    ((COMPLETED_TASKS++))
                else
                    ((FAILED_TASKS++))
                    FAILED_DATASETS+=("$dataset")
                fi
                
                unset PIDS[$pid]
                ((running_count--))
            fi
        done
        
        if [[ $running_count -ge $MAX_PARALLEL ]]; then
            sleep 1
        fi
    done
}

# 启动所有任务
task_id=0
for dataset in "${datasets[@]}"; do
    ((task_id++))
    
    wait_for_slot
    
    run_dataset "$dataset" "$task_id" &
    pid=$!
    PIDS[$pid]="$dataset"
    ((running_count++))
    
    # 稍微延迟一下，避免同时启动造成资源竞争
    sleep 1
done

# 等待所有任务完成
echo "" | tee -a "$MAIN_LOG"
echo "等待所有任务完成..." | tee -a "$MAIN_LOG"

for pid in "${!PIDS[@]}"; do
    wait "$pid"
    local exit_code=$?
    local dataset="${PIDS[$pid]}"
    
    if [[ $exit_code -eq 0 ]]; then
        ((COMPLETED_TASKS++)) || true
    else
        ((FAILED_TASKS++)) || true
        FAILED_DATASETS+=("$dataset")
    fi
done

# 显示最终统计
echo "" | tee -a "$MAIN_LOG"
echo "==========================================================" | tee -a "$MAIN_LOG"
echo "执行完成!" | tee -a "$MAIN_LOG"
echo "==========================================================" | tee -a "$MAIN_LOG"
echo "总任务数:   $TOTAL_TASKS" | tee -a "$MAIN_LOG"
echo "成功完成:   $COMPLETED_TASKS" | tee -a "$MAIN_LOG"
echo "失败任务:   $FAILED_TASKS" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

if [[ $FAILED_TASKS -gt 0 ]]; then
    echo "失败的数据集:" | tee -a "$MAIN_LOG"
    for failed_dataset in "${FAILED_DATASETS[@]}"; do
        echo "  - $failed_dataset" | tee -a "$MAIN_LOG"
    done
    echo "" | tee -a "$MAIN_LOG"
    echo "⚠️  请查看日志目录了解详情: $LOG_DIR" | tee -a "$MAIN_LOG"
    echo "==========================================================" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "✓ 所有 oner_qa 数据集处理完成！" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "预测文件位置: classifier/generate_prediction/predictions/dev_500/" | tee -a "$MAIN_LOG"
echo "现在每个数据集应该有4种检索模式的预测结果（bm25, hnsw, splade, hybrid）" | tee -a "$MAIN_LOG"
echo "==========================================================" | tee -a "$MAIN_LOG"

