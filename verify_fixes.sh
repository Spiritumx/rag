#!/bin/bash
# 验证修复是否已应用到代码中

echo "============================================================"
echo "Verifying if fixes are applied"
echo "============================================================"
echo ""

echo "1. Checking LLM server code (token-level removal)..."
echo "-----------------------------------------------------------"
grep -n "prompt_token_length = inputs.input_ids.shape\[1\]" llm_server/serve_llama.py
if [ $? -eq 0 ]; then
    echo "✓ Token-level fix found in serve_llama.py"
else
    echo "✗ Token-level fix NOT found in serve_llama.py"
    echo "  Expected line: prompt_token_length = inputs.input_ids.shape[1]"
fi

echo ""
grep -n "prompt_token_length = inputs.input_ids.shape\[1\]" llm_server/serve_llama_autobatch.py
if [ $? -eq 0 ]; then
    echo "✓ Token-level fix found in serve_llama_autobatch.py"
else
    echo "✗ Token-level fix NOT found in serve_llama_autobatch.py"
fi

echo ""
echo "2. Checking context saving code..."
echo "-----------------------------------------------------------"
grep -n "retrieved_context = {" commaqa/inference/model_search.py
if [ $? -eq 0 ]; then
    echo "✓ Context return found in model_search.py"
else
    echo "✗ Context return NOT found in model_search.py"
fi

echo ""
grep -n "contexts_file = args.output" commaqa/inference/configurable_inference.py
if [ $? -eq 0 ]; then
    echo "✓ Context saving found in configurable_inference.py"
else
    echo "✗ Context saving NOT found in configurable_inference.py"
fi

echo ""
echo "3. Checking for Python bytecode cache..."
echo "-----------------------------------------------------------"
find commaqa llm_server -name "*.pyc" -o -name "__pycache__" | head -10
echo ""
echo "To clear cache: find . -type d -name __pycache__ -exec rm -rf {} +"
echo "                find . -name '*.pyc' -delete"

echo ""
echo "4. Checking LLM server process..."
echo "-----------------------------------------------------------"
ps aux | grep -E "serve_llama|uvicorn" | grep -v grep
echo ""

echo "============================================================"
echo "Next steps:"
echo "============================================================"
echo "If fixes are missing:"
echo "  1. git pull (or sync code from local)"
echo "  2. Clear Python cache:"
echo "     find . -type d -name __pycache__ -exec rm -rf {} +"
echo "     find . -name '*.pyc' -delete"
echo "  3. Restart LLM server:"
echo "     pkill -f serve_llama"
echo "     pixi run python llm_server/serve_llama_autobatch.py &"
echo "  4. Re-run test: python evaluate/test_fixes.py"
echo ""
