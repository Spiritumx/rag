#!/bin/bash
# Start Llama Auto-Batch Inference Server with automatic request batching

# Configuration
export MODEL_PATH="/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
export LLM_PORT=8000

# Batch configuration
export BATCH_SIZE=8           # Max batch size (increase to 16 if GPU allows)
export BATCH_TIMEOUT_MS=50    # Wait time in ms (50-100ms recommended)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Starting Llama Auto-Batch Inference Server            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_PATH"
echo "Port: $LLM_PORT"
echo "Batch Size: $BATCH_SIZE"
echo "Batch Timeout: ${BATCH_TIMEOUT_MS}ms"
echo ""
echo "Auto-batching: Enabled (transparent to clients)"
echo ""

# Activate pixi environment and run
pixi run python llm_server/serve_llama_autobatch.py
