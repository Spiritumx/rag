# 并发执行脚本使用说明

## 问题背景

在修复了 `oner_qa` 的配置后，现在每个系统都需要为4种检索模式（bm25, hnsw, splade, hybrid）生成预测数据。这导致任务数量大幅增加：
- `oner_qa`: 6个数据集 × 4种检索模式 = 24个任务
- `ircot_qa`: 6个数据集 × 4种检索模式 = 24个任务

顺序执行非常耗时，因此创建了并发执行脚本来加速处理。

## 可用脚本

### 1. `run_oner_qa_only.sh` - 专门重新生成 oner_qa（推荐）

**用途**: 只重新生成 oner_qa 的预测数据，支持并发执行。

**基本用法**:
```bash
# 使用默认并发数6
./run_oner_qa_only.sh gpt 8010

# 自定义并发数（推荐2-8之间）
./run_oner_qa_only.sh gpt 8010 4

# 强制重新生成已存在的预测文件
./run_oner_qa_only.sh gpt 8010 4 --force
```

**参数说明**:
- `MODEL`: 模型名称（flan-t5-xxl, flan-t5-xl, gpt, none）
- `LLM_PORT_NUM`: LLM服务端口号（如 8010）
- `MAX_PARALLEL`: 最大并发数（可选，默认6）
- `--force`: 强制重新生成标志（可选）

**特点**:
- ✅ 自动处理所有6个数据集
- ✅ 自动为每个数据集生成4种检索模式的预测
- ✅ 详细的日志记录在 `logs_oner_qa/` 目录
- ✅ 实时显示进度和状态
- ✅ 最终统计成功和失败的任务

**日志文件**:
- 主日志: `logs_oner_qa/main_TIMESTAMP.log`
- 每个数据集日志: `logs_oner_qa/oner_qa_DATASET_TIMESTAMP.log`

---

### 2. `run_retrieval_dev_all_parallel.sh` - 通用并发执行

**用途**: 并发执行所有系统和数据集，或指定特定系统。

**基本用法**:
```bash
# 运行所有系统，默认并发数4
./run_retrieval_dev_all_parallel.sh gpt 8010

# 自定义并发数
./run_retrieval_dev_all_parallel.sh gpt 8010 8

# 只运行 oner_qa 系统
./run_retrieval_dev_all_parallel.sh gpt 8010 6 oner_qa

# 只运行 ircot_qa 系统
./run_retrieval_dev_all_parallel.sh gpt 8010 4 ircot_qa
```

**参数说明**:
- `MODEL`: 模型名称
- `LLM_PORT_NUM`: LLM服务端口号
- `MAX_PARALLEL`: 最大并发数（可选，默认4）
- `SYSTEM_FILTER`: 系统过滤器（可选，如 oner_qa）

**特点**:
- ✅ 可以运行所有系统或指定系统
- ✅ 灵活的并发控制
- ✅ 详细的任务追踪
- ✅ 日志保存在 `logs_parallel/` 目录

---

## 使用建议

### 重新生成 oner_qa 数据（你的场景）

**推荐方案**:
```bash
cd classifier/generate_prediction

# 方案1: 使用专用脚本（推荐）
./run_oner_qa_only.sh gpt 8010 4 --force

# 方案2: 使用通用并发脚本
./run_retrieval_dev_all_parallel.sh gpt 8010 6 oner_qa
```

### 并发数选择建议

根据你的硬件资源选择合适的并发数：

| 资源情况 | 推荐并发数 | 说明 |
|---------|-----------|------|
| GPU显存充足（>24GB） | 6-8 | 可以同时运行多个预测任务 |
| GPU显存一般（16-24GB） | 4-6 | 平衡速度和资源 |
| GPU显存紧张（<16GB） | 2-3 | 避免显存溢出 |
| 纯CPU执行 | 4-8 | 可以更高并发 |

**注意**: 
- 如果是 GPT API 调用，主要受限于 API 调用速率，可以使用较高并发（6-8）
- 如果是本地模型推理，主要受限于显存，需要较低并发（2-4）

### 监控执行进度

执行过程中可以通过以下方式监控：

```bash
# 实时查看主日志
tail -f logs_oner_qa/main_*.log

# 查看所有运行中的任务
ps aux | grep "runner.py"

# 查看某个数据集的详细日志
tail -f logs_oner_qa/oner_qa_nq_*.log
```

---

## 预期输出结构

成功执行后，预测文件将保存在以下结构：

```
classifier/generate_prediction/predictions/dev_500/
├── oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1___retrieval_mode__bm25/
├── oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1___retrieval_mode__hnsw/
├── oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1___retrieval_mode__splade/
├── oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1___retrieval_mode__hybrid/
├── oner_qa_gpt_trivia____prompt_set_1___bm25_retrieval_count__15___distractor_count__1___retrieval_mode__bm25/
├── ... (其他数据集和检索模式的组合)
```

每个目录下包含：
- `zero_single_multi_classification__DATASET_to_DATASET__dev_500_subsampled.json`: 预测结果

---

## 故障排除

### 问题1: 任务失败

**现象**: 有任务显示失败状态

**解决方案**:
1. 查看对应的日志文件了解错误原因
2. 检查 LLM 服务是否正常运行
3. 检查端口号是否正确
4. 对失败的数据集单独重新运行：
   ```bash
   ./run_retrieval_dev.sh oner_qa gpt FAILED_DATASET 8010
   ```

### 问题2: 显存溢出 (OOM)

**现象**: 日志中出现 "CUDA out of memory" 错误

**解决方案**:
1. 降低并发数（例如从6降到3）
2. 重新运行脚本

### 问题3: 进程卡住不动

**现象**: 任务长时间没有进展

**解决方案**:
1. 检查是否有进程僵死: `ps aux | grep python`
2. 如需要，手动终止: `Ctrl+C`
3. 清理僵尸进程: `pkill -f "runner.py"`
4. 重新运行脚本

### 问题4: 预测文件未生成检索模式后缀

**现象**: 生成的文件夹名没有 `___retrieval_mode__xxx`

**解决方案**:
这说明配置文件未正确更新，请确认：
1. `classifier/generate_prediction/run.py` 中 `oner_qa` 配置包含 `retrieval_mode`
2. 删除旧的配置文件后重新生成

---

## 性能对比

假设单个任务平均耗时 5 分钟：

| 执行方式 | 任务数 | 耗时（顺序） | 耗时（并发4） | 耗时（并发6） |
|---------|-------|-------------|--------------|--------------|
| 单个数据集 | 1 | 5分钟 | 5分钟 | 5分钟 |
| oner_qa全部 | 6 | 30分钟 | 约8分钟 | 约5分钟 |
| 所有系统 | 18 | 90分钟 | 约23分钟 | 约15分钟 |

**注意**: 实际耗时取决于：
- 模型推理速度
- API调用速率限制
- 硬件资源（GPU/CPU）
- 网络状况

---

## 下一步：生成银数据

预测数据生成完成后，可以使用以下命令生成银标签数据：

```bash
cd ../gen_sliver

# 生成银标签训练数据
python preprocess_silver_train_gpt.py --model_name gpt
```

这将自动读取所有检索模式的预测结果，并为每个问题选择最优策略。

