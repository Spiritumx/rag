# 故障排除指南

## BulkIndexError: document(s) failed to index

### 问题描述
```
elasticsearch.helpers.BulkIndexError: 1 document(s) failed to index.
```

### 原因分析

这个错误通常由以下原因引起：

#### 1. **SPLADE 向量格式问题** ⭐ 最常见
Elasticsearch 的 `rank_features` 类型对字段名有严格要求：
- ❌ 不能包含特殊 token：`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`
- ❌ 不能包含某些特殊字符
- ✅ 必须是干净的字符串

**已修复**：现在代码会自动过滤这些特殊 token。

#### 2. **Dense Embedding 维度不匹配**
- 确保索引中定义的维度与模型实际输出一致
- `all-MiniLM-L6-v2` 是 384 维
- `all-mpnet-base-v2` 是 768 维

#### 3. **内存或资源限制**
- Elasticsearch 可能内存不足
- 调整 `es_config/jvm.options.d/heap.options` 中的堆大小

### 解决方案

#### 方案 1：使用修复后的代码（推荐）
代码已经更新，会自动：
- 过滤特殊 token
- 清理 token 中的特殊字符（`#`, `##`）
- 显示详细错误信息

重新运行即可：
```bash
python retriever_server/build_index.py <dataset> --use-dense --use-splade --force
```

#### 方案 2：查看详细错误
如果还有问题，现在会显示详细错误：
```bash
python retriever_server/build_index.py <dataset> --use-dense --use-splade
```

错误信息会包含：
- 失败的文档数量
- 前 5 个错误的详细信息
- 可能的原因分析

#### 方案 3：分步测试

**测试 1：只用 BM25**
```bash
python retriever_server/build_index.py wiki
```

**测试 2：BM25 + Dense**
```bash
python retriever_server/build_index.py wiki --use-dense --force
```

**测试 3：BM25 + SPLADE**
```bash
python retriever_server/build_index.py wiki --use-splade --force
```

**测试 4：全部**
```bash
python retriever_server/build_index.py wiki --use-dense --use-splade --force
```

### 代码修改说明

#### 修改 1：SPLADE 向量清理
```python
# 修改前：直接使用 token
token = tokenizer.convert_ids_to_tokens([idx])[0]
sparse_dict[token] = float(vec_cpu[idx])

# 修改后：过滤和清理 token
token = tokenizer.convert_ids_to_tokens([int(idx)])[0]

# 过滤特殊 token
if token and not token.startswith('[') and not token.startswith('<'):
    # 清理特殊字符
    token_clean = token.replace('#', '').replace('##', '')
    if token_clean:
        sparse_dict[token_clean] = float(vec_cpu[idx])
```

#### 修改 2：详细错误信息
添加了 try-except 块来捕获并显示详细错误：
```python
try:
    result = bulk(es, ...)
except Exception as e:
    print("详细错误信息...")
    if hasattr(e, 'errors'):
        for error in e.errors[:5]:
            print(json.dumps(error, indent=2))
```

### 常见错误类型

#### 错误 1：mapper_parsing_exception
```json
{
  "type": "mapper_parsing_exception",
  "reason": "failed to parse field [splade_vector]"
}
```
**原因**：SPLADE 向量格式不正确
**解决**：使用最新代码（已修复）

#### 错误 2：illegal_argument_exception
```json
{
  "type": "illegal_argument_exception",
  "reason": "field name cannot contain special characters"
}
```
**原因**：字段名包含非法字符
**解决**：使用最新代码（已修复）

#### 错误 3：version_conflict_engine_exception
```json
{
  "type": "version_conflict_engine_exception",
  "reason": "document already exists"
}
```
**原因**：索引已存在相同 ID 的文档
**解决**：使用 `--force` 参数重建索引

### 性能优化建议

1. **增加 Elasticsearch 内存**
```bash
# 编辑 retriever_server/es_config/jvm.options.d/heap.options
-Xms1g
-Xmx1g
```

2. **减少批处理大小**
在 `build_index.py` 中添加 `chunk_size` 参数：
```python
result = bulk(
    es,
    make_documents_func(...),
    chunk_size=100,  # 默认是 500
    ...
)
```

3. **使用 GPU 加速**
```python
# 检查 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 测试checklist

在报告问题前，请确认：

- [ ] Elasticsearch 正在运行：`cd retriever_server && ./es.sh status`
- [ ] 使用最新代码（包含修复）
- [ ] 已删除旧索引：添加 `--force` 参数
- [ ] 查看了详细错误信息
- [ ] 尝试了分步测试（从只用 BM25 开始）

### 获取帮助

如果问题仍然存在，请提供：
1. 完整的错误信息（包括详细的 error JSON）
2. 使用的数据集名称
3. 使用的命令
4. Elasticsearch 版本：`curl http://localhost:9200`
5. Python 包版本：`pip list | grep -E "elasticsearch|transformers|torch"`

---

**更新日期**：2024-11
**适用版本**：Elasticsearch 8.x, Python 3.12

