"""诊断 shared cache 与 classification 的 key 匹配问题

用法: python innovation_experiments/debug_cache.py

目的: 找出为什么 Model A (cascade enabled) 的 S-Sparse EM=0.0000
      而 Model C (cascade disabled) 的 S-Sparse EM 正常
"""
import json
import os
import glob

cache_dir = "innovation_experiments/ablation_results/_shared_cache"

# ============================================================================
# 1. 列出所有缓存文件
# ============================================================================
print("=" * 70)
print("1. 缓存文件列表")
print("=" * 70)
if os.path.isdir(cache_dir):
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        size = os.path.getsize(f) / 1024
        print(f"  {os.path.basename(f):55s} {size:.1f} KB")
else:
    print(f"  缓存目录不存在: {cache_dir}")
    exit(1)

# ============================================================================
# 2. 加载 S-Sparse 缓存
# ============================================================================
print("\n" + "=" * 70)
print("2. S-Sparse 缓存内容")
print("=" * 70)

cache_file = os.path.join(cache_dir, "squad_S-Sparse_8002_inference.json")
if not os.path.exists(cache_file):
    print(f"缓存文件不存在: {cache_file}")
    print("尝试其他可能的文件名:")
    for f in glob.glob(os.path.join(cache_dir, "*squad*")):
        print(f"  找到: {os.path.basename(f)}")
    exit(1)

with open(cache_file, "r", encoding="utf-8") as f:
    cache_data = json.load(f)

cache_pred_keys = list(cache_data['predictions'].keys())
cache_ctx_keys = list(cache_data.get('contexts', {}).keys())

print(f"predictions 数量: {len(cache_pred_keys)}")
print(f"contexts 数量:    {len(cache_ctx_keys)}")
print(f"chains 数量:      {len(cache_data.get('chains', {}))}")
print(f"\n前5个 prediction keys: {cache_pred_keys[:5]}")
print(f"前5个 context keys:    {cache_ctx_keys[:5]}")

# 检查 predictions 内容
print(f"\n前3个 prediction 内容:")
for i, (qid, answer) in enumerate(cache_data['predictions'].items()):
    if i >= 3:
        break
    ans_str = answer if answer else "<EMPTY>"
    print(f"  key={qid!r}")
    print(f"  val=[{ans_str[:100]}]")

# 检查 prediction 中 "I don't know" 的数量
idk_count = sum(1 for a in cache_data['predictions'].values()
                if a.strip().lower() in ("i don't know", "i don't know.", ""))
print(f"\n'I don't know' 或空答案数量: {idk_count}/{len(cache_pred_keys)}")

# ============================================================================
# 3. 加载 classification 文件
# ============================================================================
print("\n" + "=" * 70)
print("3. Classification keys")
print("=" * 70)

# 尝试 JSONL 和 JSON 两种格式
classif_keys = None
classif_s_sparse_keys = []

for ext in ['.jsonl', '.json']:
    classif_file = f"evaluate/outputs/stage1_classifications/squad_classifications{ext}"
    if os.path.exists(classif_file):
        print(f"找到分类文件: {classif_file}")
        if ext == '.jsonl':
            classifs = {}
            with open(classif_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        classifs[item['question_id']] = item
        else:
            with open(classif_file, "r", encoding="utf-8") as f:
                classifs = json.load(f)

        classif_keys = list(classifs.keys())
        # 提取 S-Sparse 的 qids
        for qid, item in classifs.items():
            action = item.get('predicted_action', item.get('action', ''))
            if action == 'S-Sparse':
                classif_s_sparse_keys.append(qid)
        break

if classif_keys is None:
    print("!!! 分类文件不存在 !!!")
    for f in glob.glob("evaluate/outputs/stage1_classifications/squad*"):
        print(f"  找到: {f}")
else:
    print(f"总分类数量: {len(classif_keys)}")
    print(f"S-Sparse 数量: {len(classif_s_sparse_keys)}")
    print(f"\n前5个 classification keys: {classif_keys[:5]}")
    print(f"前5个 S-Sparse keys:       {classif_s_sparse_keys[:5]}")

# ============================================================================
# 4. KEY 匹配检查 (核心诊断)
# ============================================================================
print("\n" + "=" * 70)
print("4. KEY 匹配检查 (核心诊断)")
print("=" * 70)

if classif_keys is not None:
    cache_key_set = set(cache_pred_keys)
    classif_key_set = set(classif_s_sparse_keys)

    common = cache_key_set & classif_key_set
    only_in_cache = cache_key_set - classif_key_set
    only_in_classif = classif_key_set - cache_key_set

    print(f"缓存 predictions keys:  {len(cache_key_set)}")
    print(f"分类 S-Sparse keys:     {len(classif_key_set)}")
    print(f"交集 (匹配):            {len(common)}")
    print(f"仅在缓存中:             {len(only_in_cache)}")
    print(f"仅在分类中:             {len(only_in_classif)}")

    if len(common) == 0 and len(cache_key_set) > 0 and len(classif_key_set) > 0:
        print("\n!!! KEY 完全不匹配 — 这就是 EM=0 的根本原因 !!!")
        print("\n示例对比:")
        sample_cache = list(only_in_cache)[:3]
        sample_classif = list(only_in_classif)[:3]
        for c, cl in zip(sample_cache, sample_classif):
            print(f"  cache key:  {c!r}  (type={type(c).__name__}, len={len(c)})")
            print(f"  classif key:{cl!r}  (type={type(cl).__name__}, len={len(cl)})")
            # 检查是否有公共子串
            if c in cl:
                print(f"  => cache key 是 classif key 的子串!")
            elif cl in c:
                print(f"  => classif key 是 cache key 的子串!")
            print()
    elif len(common) < len(classif_key_set):
        print(f"\n部分匹配 — {len(classif_key_set) - len(common)} 个 S-Sparse 问题在缓存中找不到")
        missing_examples = list(only_in_classif)[:3]
        print(f"缺失示例: {missing_examples}")
    else:
        print("\n✓ 所有 S-Sparse 分类 keys 在缓存中都有匹配")

    # 额外检查: context keys 是否也匹配
    cache_ctx_set = set(cache_ctx_keys)
    ctx_common = cache_ctx_set & classif_key_set
    print(f"\ncontext keys 与 S-Sparse keys 交集: {len(ctx_common)}")

# ============================================================================
# 5. Model A 最终预测
# ============================================================================
print("\n" + "=" * 70)
print("5. Model A 最终预测")
print("=" * 70)

pred_file_a = "innovation_experiments/ablation_results/Model_A_Full/stage2_predictions/squad_predictions_v2.json"
if os.path.exists(pred_file_a):
    with open(pred_file_a, "r", encoding="utf-8") as f:
        preds_a = json.load(f)
    print(f"预测数量: {len(preds_a)}")
    print(f"前5个 keys: {list(preds_a.keys())[:5]}")

    # 统计 "I don't know" 数量
    idk_a = sum(1 for a in preds_a.values()
                if isinstance(a, str) and a.strip().lower() in ("i don't know", "i don't know.", ""))
    print(f"'I don't know' 数量: {idk_a}/{len(preds_a)}")

    # 如果有 S-Sparse keys, 检查 S-Sparse 预测
    if classif_s_sparse_keys:
        s_sparse_preds = {qid: preds_a.get(qid, "<MISSING>") for qid in classif_s_sparse_keys[:5]}
        print(f"\nS-Sparse 前5个预测:")
        for qid, ans in s_sparse_preds.items():
            print(f"  {qid}: [{ans[:80] if isinstance(ans, str) else ans}]")

        idk_s = sum(1 for qid in classif_s_sparse_keys
                    if qid in preds_a and isinstance(preds_a[qid], str)
                    and preds_a[qid].strip().lower() in ("i don't know", "i don't know.", ""))
        print(f"\nS-Sparse 中 'I don't know' 数量: {idk_s}/{len(classif_s_sparse_keys)}")
else:
    print(f"Model A 预测文件不存在: {pred_file_a}")
    # 尝试不带 _v2 后缀
    alt = pred_file_a.replace("_v2.json", ".json")
    if os.path.exists(alt):
        print(f"  但找到: {alt}")

# ============================================================================
# 6. Model C 最终预测
# ============================================================================
print("\n" + "=" * 70)
print("6. Model C 最终预测")
print("=" * 70)

pred_file_c = "innovation_experiments/ablation_results/Model_C_wo_Cascade/stage2_predictions/squad_predictions_v2.json"
if os.path.exists(pred_file_c):
    with open(pred_file_c, "r", encoding="utf-8") as f:
        preds_c = json.load(f)
    print(f"预测数量: {len(preds_c)}")
    print(f"前5个 keys: {list(preds_c.keys())[:5]}")

    # 统计 "I don't know" 数量
    idk_c = sum(1 for a in preds_c.values()
                if isinstance(a, str) and a.strip().lower() in ("i don't know", "i don't know.", ""))
    print(f"'I don't know' 数量: {idk_c}/{len(preds_c)}")

    # Model A vs Model C key 对比
    if os.path.exists(pred_file_a):
        a_keys = set(preds_a.keys())
        c_keys = set(preds_c.keys())
        print(f"\nModel A keys ∩ Model C keys: {len(a_keys & c_keys)}")
        print(f"仅在 A: {len(a_keys - c_keys)}")
        print(f"仅在 C: {len(c_keys - a_keys)}")

        # 对比共同 key 的 prediction 是否相同
        common_keys = a_keys & c_keys
        if common_keys:
            same_count = sum(1 for k in common_keys if preds_a[k] == preds_c[k])
            print(f"共同 key 中预测相同: {same_count}/{len(common_keys)}")

            # 找一个不同的例子
            for k in list(common_keys)[:10]:
                if preds_a[k] != preds_c[k]:
                    print(f"\n差异示例 (key={k}):")
                    print(f"  Model A: [{preds_a[k][:80]}]")
                    print(f"  Model C: [{preds_c[k][:80]}]")
                    break
else:
    print(f"Model C 预测文件不存在: {pred_file_c}")

# ============================================================================
# 7. Test data key 格式
# ============================================================================
print("\n" + "=" * 70)
print("7. Test data key 格式")
print("=" * 70)

test_file = "processed_data/squad/test_subsampled.jsonl"
if os.path.exists(test_file):
    test_keys = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_keys.append(item['question_id'])
    print(f"test data 数量: {len(test_keys)}")
    print(f"前5个 keys: {test_keys[:5]}")

    # 对比 test data keys 和 cache keys
    test_set = set(test_keys)
    cache_set = set(cache_pred_keys)
    print(f"\ntest keys ∩ cache keys: {len(test_set & cache_set)}")
    print(f"仅在 test: {len(test_set - cache_set)}")
    print(f"仅在 cache: {len(cache_set - test_set)}")
else:
    print(f"test data 不存在: {test_file}")

# ============================================================================
# 8. 总结
# ============================================================================
print("\n" + "=" * 70)
print("8. 诊断总结")
print("=" * 70)

issues = []
if classif_keys is not None and len(common) == 0 and len(cache_key_set) > 0:
    issues.append("CRITICAL: Cache keys 与 Classification keys 完全不匹配")
if os.path.exists(pred_file_a):
    if idk_a > len(preds_a) * 0.5:
        issues.append(f"CRITICAL: Model A 有 {idk_a}/{len(preds_a)} 个 'I don't know' 回答")
if not issues:
    issues.append("未发现明显问题，需要进一步检查")

for issue in issues:
    print(f"  => {issue}")

print("\n完成。请将输出贴回对话。")
