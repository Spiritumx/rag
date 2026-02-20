#!/bin/bash
# =============================================================================
# 重新运行实验脚本 - ToT 提前终止修复后
#
# 只跑消融实验 A, C, D (B 用线性 M_core，不受影响)
# Model A = V2 完整系统，跑完后复制到 V2 输出目录
#
# 前置条件:
#   - LLM server 运行在 port 8000
#   - Baseline retriever 运行在 port 8001 (Model D 需要)
#   - V2 retriever 运行在 port 8002 (Model A, C 需要)
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  ToT Fix 后重新运行实验"
echo "  项目目录: $PROJECT_ROOT"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ============================================================================
# Step 1: 备份旧结果
# ============================================================================
echo ""
echo "[Step 1] 备份旧结果..."

ABLATION_DIR="innovation_experiments/ablation_results"
V2_OUTPUT="innovation_experiments/evaluate_v2/outputs_v2"

# 备份消融 A, C, D
for model in Model_A_Full Model_C_wo_Cascade Model_D_wo_Adaptive; do
    if [ -d "$ABLATION_DIR/$model" ]; then
        echo "  备份 $model"
        mv "$ABLATION_DIR/$model" "$ABLATION_DIR/${model}_backup_${TIMESTAMP}"
    fi
done

# 备份 V2 主输出
for subdir in stage2_predictions_v2 stage3_metrics_v2 cascade_analysis; do
    if [ -d "$V2_OUTPUT/$subdir" ]; then
        echo "  备份 V2 $subdir"
        mv "$V2_OUTPUT/$subdir" "$V2_OUTPUT/${subdir}_backup_${TIMESTAMP}"
    fi
done

# ============================================================================
# Step 2: 跑消融实验 A, C, D
# ============================================================================
echo ""
echo "============================================================"
echo "[Step 2] 运行消融实验 A, C, D (跳过 B)"
echo "============================================================"

python innovation_experiments/run_ablation_experiments.py \
    --experiments A C D \
    --datasets squad hotpotqa trivia nq musique 2wikimultihopqa \
    --stages 2 3

# ============================================================================
# Step 3: 把 Model A 结果复制到 V2 主输出目录
# ============================================================================
echo ""
echo "[Step 3] 复制 Model A 结果 → V2 主输出目录..."

MODEL_A="$ABLATION_DIR/Model_A_Full"

if [ -d "$MODEL_A/stage2_predictions" ]; then
    cp -r "$MODEL_A/stage2_predictions" "$V2_OUTPUT/stage2_predictions_v2"
    echo "  stage2 ✓"
fi

if [ -d "$MODEL_A/stage3_metrics" ]; then
    cp -r "$MODEL_A/stage3_metrics" "$V2_OUTPUT/stage3_metrics_v2"
    echo "  stage3 ✓"
fi

if [ -d "$MODEL_A/cascade_analysis" ]; then
    cp -r "$MODEL_A/cascade_analysis" "$V2_OUTPUT/cascade_analysis"
    echo "  cascade ✓"
fi

# ============================================================================
# Step 4: 对比全部消融结果
# ============================================================================
echo ""
echo "============================================================"
echo "[Step 4] 对比全部消融结果 (A, B, C, D)"
echo "============================================================"

python innovation_experiments/run_ablation_experiments.py --compare-only

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "============================================================"
echo "  全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  V2 指标:     $V2_OUTPUT/stage3_metrics_v2/overall_metrics_v2.json"
echo "  消融对比:    $ABLATION_DIR/ablation_comparison.json"
echo ""
echo "下一步: 用新数据更新论文图表"
echo "  - generate_plots.py Plot 13 的 v2_retr"
echo "  - chapter04.tex tab:chapter4_retrieval_cost"
echo "  - chapter04.tex tab:chapter4_ablation_results"
