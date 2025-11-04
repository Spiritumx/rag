# 🚀 Quick Reference Card

## 两步使用

```bash
# 第 1 步：下载模型（只需一次）
cd retriever_server && python download_model.py --all && cd ..

# 第 2 步：构建索引（自动使用本地模型）
python retriever_server/build_index.py <dataset> --use-dense --use-splade
```

## 常用命令

| 功能 | 命令 |
|------|------|
| **下载所有模型** | `cd retriever_server && python download_model.py --all` |
| **只下载 Dense** | `cd retriever_server && python download_model.py --download-dense` |
| **只下载 SPLADE** | `cd retriever_server && python download_model.py --download-splade` |
| **只用 BM25** | `python retriever_server/build_index.py <dataset>` |
| **BM25 + Dense** | `python retriever_server/build_index.py <dataset> --use-dense` |
| **BM25 + SPLADE** | `python retriever_server/build_index.py <dataset> --use-splade` |
| **三种全用** | `python retriever_server/build_index.py <dataset> --use-dense --use-splade` |
| **强制重建** | 添加 `--force` 参数 |

## Elasticsearch 管理

```bash
cd retriever_server
./es.sh install    # 安装
./es.sh start      # 启动
./es.sh status     # 状态
./es.sh stop       # 停止
```

## 支持的数据集

- `hotpotqa`
- `iirc`
- `2wikimultihopqa`
- `musique`
- `wiki`
- `nq`
- `trivia`
- `squad`

## 目录结构

```
retriever_server/
├── models/                          # ← 模型自动存储在这里
│   ├── dense/all-MiniLM-L6-v2/
│   └── splade/splade-cocondenser-ensembledistil/
├── download_model.py                # ← 下载工具
└── build_index.py                   # ← 构建工具（自动加载）
```

## 自动检测

✅ `build_index.py` 会自动从 `retriever_server/models/` 加载模型  
✅ 不需要手动指定路径  
✅ 下载一次，永久使用

## 推荐配置

| 场景 | 命令 |
|------|------|
| **快速测试** | `python retriever_server/build_index.py wiki` |
| **日常使用** | `python retriever_server/build_index.py wiki --use-splade` |
| **最佳效果** | `python retriever_server/build_index.py wiki --use-dense --use-splade` |

## 环境变量（可选）

```bash
# 使用 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com
```

---

📖 详细文档：`SIMPLE_GUIDE.md`

