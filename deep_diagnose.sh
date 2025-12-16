#!/bin/bash
# 深度诊断：检查为什么修复没有生效

echo "============================================================"
echo "Deep Diagnosis: Why fixes are not working"
echo "============================================================"
echo ""

echo "1. Clearing ALL Python cache (not just commaqa/llm_server)..."
echo "-----------------------------------------------------------"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
echo "✓ All Python cache cleared"

echo ""
echo "2. Checking if contexts are returned in code..."
echo "-----------------------------------------------------------"
echo "Checking model_search.py for 4-tuple return:"
grep -A 3 "retrieved_context = {" commaqa/inference/model_search.py | head -5

echo ""
echo "3. Testing LLM server directly..."
echo "-----------------------------------------------------------"
echo "Testing with a simple prompt to see if truncation happens..."
RESPONSE=$(curl -s "http://localhost:8000/generate?prompt=The%20capital%20of%20France%20is%20Paris&max_length=10&keep_prompt=false")
echo "Response: $RESPONSE"
echo ""
echo "If you see truncation in the response above, the server fix didn't work."

echo ""
echo "4. Checking actual prediction file..."
echo "-----------------------------------------------------------"
echo "First 3 predictions from squad_predictions.json:"
head -20 evaluate/outputs/stage2_predictions/squad_predictions.json

echo ""
echo "5. Checking chains file for LLM interaction..."
echo "-----------------------------------------------------------"
echo "First chain (should show LLM prompts and responses):"
head -50 evaluate/outputs/stage2_predictions/squad_predictions_chains.txt

echo ""
echo "6. Checking if inference is calling the new code..."
echo "-----------------------------------------------------------"
echo "Testing return_qid_prediction signature:"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from commaqa.inference.model_search import BestFirstDecomposer
import inspect

sig = inspect.signature(BestFirstDecomposer.return_qid_prediction)
print(f"Method signature: {sig}")

# Try to check the source
source = inspect.getsource(BestFirstDecomposer.return_qid_prediction)
if "retrieved_context" in source:
    print("✓ New code is loaded (has retrieved_context)")
else:
    print("✗ Old code is loaded (no retrieved_context)")
EOF

echo ""
echo "============================================================"
echo "Diagnosis complete"
echo "============================================================"
