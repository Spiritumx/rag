#!/bin/bash
# 自动化训练流程脚本
# 按顺序执行: finetune.py -> merge_model.py -> evaluate.py

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 记录开始时间
START_TIME=$(date +%s)

echo "======================================================================="
echo "🚀  自动化训练流程"
echo "======================================================================="
echo "📂 工作目录: $PROJECT_ROOT"
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 函数: 执行单个脚本
run_script() {
    local step=$1
    local total=$2
    local title=$3
    local script=$4

    echo ""
    echo "======================================================================="
    echo -e "${BLUE}  [$step/$total] $title${NC}"
    echo "======================================================================="
    echo "📝 执行: python $script"
    echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    step_start=$(date +%s)

    # 执行脚本
    if python "$SCRIPT_DIR/$script"; then
        step_end=$(date +%s)
        step_duration=$((step_end - step_start))
        echo ""
        echo -e "${GREEN}✅ $title 完成！${NC}"
        echo "⏱️  耗时: $step_duration 秒"
    else
        step_end=$(date +%s)
        step_duration=$((step_end - step_start))
        echo ""
        echo -e "${RED}❌ $title 失败！${NC}"
        echo "⏱️  运行时长: $step_duration 秒"
        exit 1
    fi
}

# 执行流程
run_script 1 3 "微调训练" "finetune.py"
run_script 2 3 "合并模型" "merge_model.py"
run_script 3 3 "模型评估" "evaluate.py"

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "======================================================================="
echo -e "${GREEN}🎉  所有步骤执行成功！${NC}"
echo "======================================================================="
echo "⏱️  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "⏰ 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 流程总结:"
echo "  ✅ [1] 微调训练"
echo "  ✅ [2] 合并模型"
echo "  ✅ [3] 模型评估"
echo ""
