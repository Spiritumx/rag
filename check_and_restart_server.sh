#!/bin/bash
# 检查并重启LLM服务器

echo "============================================================"
echo "Checking and Restarting LLM Server"
echo "============================================================"
echo ""

echo "Step 1: Stopping existing LLM server..."
echo "-----------------------------------------------------------"
pkill -f "serve_llama"
sleep 2

# Verify stopped
if ps aux | grep -E "serve_llama|uvicorn.*8000" | grep -v grep > /dev/null; then
    echo "⚠️  Server still running, force killing..."
    pkill -9 -f "serve_llama"
    sleep 1
fi

if ps aux | grep -E "serve_llama|uvicorn.*8000" | grep -v grep > /dev/null; then
    echo "✗ Failed to stop server"
    exit 1
else
    echo "✓ Server stopped"
fi

echo ""
echo "Step 2: Clearing Python cache..."
echo "-----------------------------------------------------------"
find commaqa llm_server -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find commaqa llm_server -name '*.pyc' -delete 2>/dev/null
echo "✓ Cache cleared"

echo ""
echo "Step 3: Verifying fixes are in code..."
echo "-----------------------------------------------------------"

# Check for token-level fix
if grep -q "prompt_token_length = inputs.input_ids.shape\[1\]" llm_server/serve_llama_autobatch.py; then
    echo "✓ Token-level fix found in serve_llama_autobatch.py"
else
    echo "✗ Token-level fix NOT found in serve_llama_autobatch.py"
    echo "  Please sync the latest code first!"
    exit 1
fi

# Check for dual route
if grep -q '@app.get("/generate")' llm_server/serve_llama_autobatch.py && \
   grep -q '@app.get("/generate/")' llm_server/serve_llama_autobatch.py; then
    echo "✓ Dual route fix found"
else
    echo "✗ Dual route fix NOT found"
    echo "  Please sync the latest code first!"
    exit 1
fi

echo ""
echo "Step 4: Starting LLM server..."
echo "-----------------------------------------------------------"
echo "Using autobatch server for better performance"

# Start in background
nohup pixi run python llm_server/serve_llama_autobatch.py > llm_server.log 2>&1 &
SERVER_PID=$!

echo "Server starting (PID: $SERVER_PID)..."
echo "Waiting for server to be ready..."

# Wait for server to start (max 60 seconds)
for i in {1..60}; do
    sleep 1
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        echo ""
        echo "Server info:"
        curl -s http://localhost:8000/ | python -m json.tool
        echo ""
        echo "Log file: llm_server.log"
        echo "To view logs: tail -f llm_server.log"
        exit 0
    fi
    echo -n "."
done

echo ""
echo "✗ Server failed to start within 60 seconds"
echo "Check logs: tail -20 llm_server.log"
exit 1
