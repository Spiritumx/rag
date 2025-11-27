#!/bin/bash

# 目标基础路径
TARGET_BASE="/autodl-tmp"

# 检查是否在 AutoDL 环境
if [ ! -d "$TARGET_BASE" ]; then
    echo "Warning: $TARGET_BASE does not exist."
    read -p "Are you running this on the AutoDL server? Continue anyway? (y/n): " response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi

# 获取当前工作目录
PROJECT_ROOT=$(pwd)

# 源模型路径
SRC_MODEL_DIR="$PROJECT_ROOT/classifier/model/Qwen2.5-3B-Instruct"

# 目标路径
TARGET_MODEL_DIR="$TARGET_BASE/model/Qwen2.5-3B-Instruct"
TARGET_DATA_DIR="$TARGET_BASE/data"
TARGET_OUTPUT_DIR="$TARGET_BASE/output"

echo "Source Model: $SRC_MODEL_DIR"
echo "Target Model: $TARGET_MODEL_DIR"

# 1. 迁移模型
if [ -d "$SRC_MODEL_DIR" ]; then
    if [ -d "$TARGET_MODEL_DIR" ]; then
        echo "Target directory $TARGET_MODEL_DIR already exists. Skipping copy."
    else
        echo "Copying model files... This may take a while."
        # 确保目标父目录存在
        mkdir -p "$(dirname "$TARGET_MODEL_DIR")"
        
        # 复制目录
        cp -r "$SRC_MODEL_DIR" "$TARGET_MODEL_DIR"
        
        if [ $? -eq 0 ]; then
            echo "Model copied successfully."
        else
            echo "Error copying model."
            exit 1
        fi
    fi
else
    echo "Warning: Source model directory not found at $SRC_MODEL_DIR"
fi

# 2. 创建数据和输出目录
echo "Creating/Verifying data directory: $TARGET_DATA_DIR"
mkdir -p "$TARGET_DATA_DIR"

echo "Creating/Verifying output directory: $TARGET_OUTPUT_DIR"
mkdir -p "$TARGET_OUTPUT_DIR"

echo ""
echo "Migration setup complete."
echo "Please verify the files in /autodl-tmp/"

