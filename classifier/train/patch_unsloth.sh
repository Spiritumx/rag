#!/bin/bash
# 直接修改 unsloth 源码，绕过 logits 检查

UTILS_FILE="/root/graduateRAG/.pixi/envs/default/lib/python3.12/site-packages/unsloth/models/_utils.py"

if [ ! -f "$UTILS_FILE" ]; then
    echo "错误: 找不到文件 $UTILS_FILE"
    exit 1
fi

echo "正在备份原文件..."
cp "$UTILS_FILE" "${UTILS_FILE}.backup"

echo "正在修改 _utils.py..."

# 方法1: 将 raise_logits_error 函数改为直接返回，不抛出错误
sed -i 's/raise NotImplementedError(LOGITS_ERROR_STRING)/pass  # Patched: skip logits check/g' "$UTILS_FILE"

# 方法2: 强制设置 UNSLOTH_RETURN_LOGITS 为 True
sed -i 's/UNSLOTH_RETURN_LOGITS = False/UNSLOTH_RETURN_LOGITS = True/g' "$UTILS_FILE"
sed -i 's/UNSLOTH_RETURN_LOGITS = None/UNSLOTH_RETURN_LOGITS = True/g' "$UTILS_FILE"

# 方法3: 修改环境变量检查逻辑
sed -i 's/os\.environ\.get.*UNSLOTH_RETURN_LOGITS.*==.*1.*/True/g' "$UTILS_FILE"

echo "✓ 修改完成！"
echo "原文件已备份到: ${UTILS_FILE}.backup"
echo ""
echo "如果需要恢复，运行: cp ${UTILS_FILE}.backup $UTILS_FILE"
