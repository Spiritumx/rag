#!/bin/bash
# Start Llama Batch Inference Server

# Configuration
export MODEL_PATH="/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
export LLM_PORT=8000

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Starting Llama Batch Inference Server              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_PATH"
echo "Port: $LLM_PORT"
echo ""

# Activate pixi environment and run
pixi run python llm_server/serve_llama.py
