#!/bin/bash

# 解压数据集脚本
# 将 /root/autodl-fs 目录下的 .7z 和 .zip 文件解压到 download/raw_data 目录

set -e  # 遇到错误时退出

# 定义源目录和目标目录
SOURCE_DIR="/root/autodl-fs"
TARGET_DIR="/root/autodl-tmp/raw_data"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

echo "开始解压文件..."
echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo ""

# 解压所有 .7z 文件
echo "正在解压 .7z 文件..."
for file in "$SOURCE_DIR"/*.7z; do
    if [ -f "$file" ]; then
        echo "解压: $(basename "$file")"
        7z x "$file" -o"$TARGET_DIR" -y
    fi
done

echo ""

# 解压所有 .zip 文件
echo "正在解压 .zip 文件..."
for file in "$SOURCE_DIR"/*.zip; do
    if [ -f "$file" ]; then
        echo "解压: $(basename "$file")"
        unzip -q -o "$file" -d "$TARGET_DIR"
    fi
done

echo ""
echo "解压完成！"
echo "所有文件已解压到: $TARGET_DIR"
