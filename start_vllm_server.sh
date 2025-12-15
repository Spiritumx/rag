#!/bin/bash
# Start vLLM server for high-throughput LLM inference

# Configuration
export LLM_MODEL_PATH="/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
export LLM_PORT=8000
export TENSOR_PARALLEL_SIZE=1  # Set to 2 for multi-GPU (if you have 2 GPUs)
export GPU_MEMORY_UTILIZATION=0.85  # Use 85% of GPU memory

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Starting vLLM Server                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $LLM_MODEL_PATH"
echo "Port: $LLM_PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "GPU Memory: ${GPU_MEMORY_UTILIZATION}%"
echo ""

# Activate pixi environment and run
pixi run python evaluate/utils/vllm_server.py
