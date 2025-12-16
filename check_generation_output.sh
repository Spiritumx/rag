#!/bin/bash
# 检查生成输出文件

echo "============================================================"
echo "Checking Generation Output Files"
echo "============================================================"
echo ""

OUTPUT_DIR="evaluate/outputs/stage2_predictions"

echo "1. Files in output directory:"
echo "-----------------------------------------------------------"
ls -lht $OUTPUT_DIR/squad_predictions* 2>/dev/null || echo "No files found!"
echo ""

echo "2. File timestamps (to see if they're from the recent run):"
echo "-----------------------------------------------------------"
stat $OUTPUT_DIR/squad_predictions.json 2>/dev/null || echo "predictions.json not found"
stat $OUTPUT_DIR/squad_predictions_chains.txt 2>/dev/null || echo "chains.txt not found"
stat $OUTPUT_DIR/squad_predictions_contexts.json 2>/dev/null || echo "contexts.json not found"
echo ""

echo "3. Size of predictions file:"
echo "-----------------------------------------------------------"
wc -l $OUTPUT_DIR/squad_predictions.json 2>/dev/null || echo "predictions.json not found"
echo ""

echo "4. Checking if this is using cached/old results:"
echo "-----------------------------------------------------------"
echo "Current time:"
date
echo ""
echo "Last modified time of predictions file:"
ls -l $OUTPUT_DIR/squad_predictions.json 2>/dev/null | awk '{print $6, $7, $8}'
echo ""

echo "5. Check if generation actually ran or used cache:"
echo "-----------------------------------------------------------"
echo "Looking for time_taken file:"
cat $OUTPUT_DIR/squad_predictions_time_taken.txt 2>/dev/null || echo "time_taken.txt not found"
echo ""

echo "6. Check full_eval_path (shows what input was used):"
echo "-----------------------------------------------------------"
cat $OUTPUT_DIR/squad_predictions_full_eval_path.txt 2>/dev/null || echo "full_eval_path.txt not found"
echo ""

echo "7. Checking if there's a separate output directory for actions:"
echo "-----------------------------------------------------------"
find evaluate/outputs -name "*squad*" -type f 2>/dev/null | head -20
echo ""

echo "8. Check if stage2_generate.py has multiple output locations:"
echo "-----------------------------------------------------------"
grep -n "predictions.json\|_chains.txt\|_contexts.json" evaluate/stage2_generate.py | head -20
echo ""

echo "============================================================"
echo "If chains.txt doesn't exist, possible reasons:"
echo "============================================================"
echo "1. Generation didn't actually run (used cached results)"
echo "2. Generation crashed before saving chains"
echo "3. Chains are being saved to a different location"
echo "4. The inference code has a bug preventing chain saving"
echo ""
