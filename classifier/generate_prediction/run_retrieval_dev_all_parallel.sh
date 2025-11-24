#!/usr/bin/env bash

# This script runs predictions in parallel for faster execution.
# It supports concurrent execution with configurable parallelism.

set -euo pipefail

systems=("ircot_qa" "oner_qa" "nor_qa")
valid_models=("flan-t5-xxl" "flan-t5-xl" "gpt" "none")
datasets=("hotpotqa" "2wikimultihopqa" "musique" "nq" "trivia" "squad")

usage() {
    echo "Usage: $0 MODEL LLM_PORT_NUM [MAX_PARALLEL] [SYSTEM_FILTER]"
    echo "  MODEL         : ${valid_models[*]}"
    echo "  LLM_PORT_NUM  : Port of the LLM service (e.g., 8010)"
    echo "  MAX_PARALLEL  : Maximum parallel jobs (default: 4)"
    echo "  SYSTEM_FILTER : Only run specific system (optional, e.g., 'oner_qa')"
    echo ""
    echo "Example:"
    echo "  $0 gpt 8010 4 oner_qa    # Run only oner_qa with 4 parallel jobs"
    echo "  $0 gpt 8010 2            # Run all systems with 2 parallel jobs"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

MODEL="$1"
LLM_PORT="$2"
MAX_PARALLEL="${3:-4}"  # 默认并发数为4
SYSTEM_FILTER="${4:-}"  # 可选的系统过滤器

# Validation helpers
check_in_list() {
    local value="$1"
    shift
    local list=("$@")
    for item in "${list[@]}"; do
        if [[ "$item" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

if ! check_in_list "$MODEL" "${valid_models[@]}"; then
    echo "Invalid MODEL: $MODEL. Expected one of: ${valid_models[*]}"
    usage
fi

if [[ -n "$SYSTEM_FILTER" ]] && ! check_in_list "$SYSTEM_FILTER" "${systems[@]}"; then
    echo "Invalid SYSTEM_FILTER: $SYSTEM_FILTER. Expected one of: ${systems[*]}"
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_parallel"
mkdir -p "$LOG_DIR"

# 记录任务信息
TASK_FILE="${LOG_DIR}/tasks_$(date +%Y%m%d_%H%M%S).txt"
: > "$TASK_FILE"  # 清空文件

# 统计变量
TOTAL_TASKS=0
RUNNING_TASKS=0
COMPLETED_TASKS=0
FAILED_TASKS=0

# 存储后台任务的PID和信息
declare -A TASK_PIDS
declare -A TASK_INFO

# 执行单个任务的函数
run_task() {
    local system="$1"
    local dataset="$2"
    local task_id="$3"
    
    local log_file="${LOG_DIR}/${system}_${dataset}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$task_id] Starting: $system | $dataset" | tee -a "$TASK_FILE"
    
    # 执行任务并记录日志
    if "${SCRIPT_DIR}/run_retrieval_dev.sh" "$system" "$MODEL" "$dataset" "$LLM_PORT" > "$log_file" 2>&1; then
        echo "[$task_id] ✓ Completed: $system | $dataset" | tee -a "$TASK_FILE"
        return 0
    else
        echo "[$task_id] ✗ Failed: $system | $dataset (see $log_file)" | tee -a "$TASK_FILE"
        return 1
    fi
}

# 等待直到有空闲槽位
wait_for_slot() {
    while [[ $RUNNING_TASKS -ge $MAX_PARALLEL ]]; do
        # 检查是否有任务完成
        for pid in "${!TASK_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # 任务已完成
                wait "$pid"
                local exit_code=$?
                
                if [[ $exit_code -eq 0 ]]; then
                    ((COMPLETED_TASKS++))
                else
                    ((FAILED_TASKS++))
                fi
                
                unset TASK_PIDS[$pid]
                ((RUNNING_TASKS--))
            fi
        done
        
        # 如果还是满的，等待一下
        if [[ $RUNNING_TASKS -ge $MAX_PARALLEL ]]; then
            sleep 2
        fi
    done
}

# 构建任务列表
echo ""
echo "=========================================================="
echo "并发执行配置:"
echo "  模型: $MODEL"
echo "  LLM端口: $LLM_PORT"
echo "  最大并发数: $MAX_PARALLEL"
if [[ -n "$SYSTEM_FILTER" ]]; then
    echo "  系统过滤: $SYSTEM_FILTER"
fi
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
echo ""

# 收集所有需要执行的任务
tasks_to_run=()

for system in "${systems[@]}"; do
    # 应用系统过滤器
    if [[ -n "$SYSTEM_FILTER" ]] && [[ "$system" != "$SYSTEM_FILTER" ]]; then
        continue
    fi
    
    if [[ "$MODEL" == "none" && "$system" != "oner" ]]; then
        echo "Skipping system $system because MODEL 'none' is only valid with 'oner'."
        continue
    fi

    for dataset in "${datasets[@]}"; do
        tasks_to_run+=("$system:$dataset")
        ((TOTAL_TASKS++))
    done
done

echo "总任务数: $TOTAL_TASKS"
echo ""

# 执行任务
task_counter=0
for task in "${tasks_to_run[@]}"; do
    IFS=':' read -r system dataset <<< "$task"
    ((task_counter++))
    
    # 等待空闲槽位
    wait_for_slot
    
    # 启动新任务
    run_task "$system" "$dataset" "$task_counter" &
    local pid=$!
    TASK_PIDS[$pid]=1
    TASK_INFO[$pid]="$system:$dataset"
    ((RUNNING_TASKS++))
    
    echo "[主进程] 已启动任务 $task_counter/$TOTAL_TASKS: $system | $dataset (PID: $pid)"
done

# 等待所有任务完成
echo ""
echo "等待所有任务完成..."
wait

# 更新最终统计
for pid in "${!TASK_PIDS[@]}"; do
    wait "$pid"
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        ((COMPLETED_TASKS++))
    else
        ((FAILED_TASKS++))
    fi
done

# 显示最终统计
echo ""
echo "=========================================================="
echo "执行完成!"
echo "=========================================================="
echo "总任务数:   $TOTAL_TASKS"
echo "成功完成:   $COMPLETED_TASKS"
echo "失败任务:   $FAILED_TASKS"
echo "日志目录:   $LOG_DIR"
echo "任务记录:   $TASK_FILE"
echo "=========================================================="

if [[ $FAILED_TASKS -gt 0 ]]; then
    echo ""
    echo "⚠️  有 $FAILED_TASKS 个任务失败，请查看日志文件了解详情。"
    exit 1
fi

echo ""
echo "✓ 所有任务成功完成！"


