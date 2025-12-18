# Testing Instructions for Enhanced System Prompt

## Changes Made

Updated `llm_server/serve_llama_autobatch.py` (Lines 82-119) with enhanced system prompt that includes:

1. **Branch 4**: Explicit "I don't know" guidance after 8+ failed attempts
2. **Entity Requirements**: Clear examples of good vs bad sentences
3. **Entity Checklist**: Three-point checklist before generation
4. **Better Organization**: Separated OUTPUT FORMAT from DECISION LOGIC

## Testing Options

### Option 1: Full Dataset Re-evaluation (Recommended)

Since the system is already set up and you have existing baselines, run full evaluation:

```bash
# Navigate to project directory
cd D:\code\graduate\graduateRAG

# Run Stage 2 generation for musique (M strategy will use updated prompt)
python evaluate/stage2_generate.py --config evaluate/config.yaml --datasets musique

# Run Stage 2 for 2wikimultihopqa
python evaluate/stage2_generate.py --config evaluate/config.yaml --datasets 2wikimultihopqa

# Evaluate results
python evaluate/stage3_evaluate.py --config evaluate/config.yaml --datasets musique 2wikimultihopqa

# Analyze with diagnostic tool
python analyze_ircot_detailed.py evaluate/outputs/stage2_predictions
```

**Expected Time:**
- musique: ~2.5 hours (500 questions, M strategy ~5 min/question with parallelism)
- 2wikimultihopqa: ~2.5 hours

### Option 2: Manual Subset Testing (Faster)

If you want to test on ~100 samples first:

1. **Create subset data files:**

```bash
# Create a temporary test subset
cd D:\code\graduate\graduateRAG\data

# Take first 100 lines from musique test
head -100 musique/musique_test.jsonl > musique/musique_test_sample.jsonl
```

2. **Temporarily update config to use subset:**

Edit `evaluate/config.yaml`:
```yaml
datasets:
  - musique_test_sample  # Use subset instead of musique
```

3. **Run pipeline:**
```bash
python evaluate/run_pipeline.py --config evaluate/config.yaml
```

4. **Restore config** after testing

### Option 3: Direct Inference Testing (Quickest Validation)

Test the prompt on a single question to verify it works:

```bash
# Start LLM server if not running
# (Assuming it's already running based on your setup)

# Use commaqa CLI to test a single question
python -m commaqa.inference.configurable_inference \
  --config evaluate/configs/llama_configs/multi_hop.jsonnet \
  --input data/musique/musique_test.jsonl \
  --output test_output.json
```

Then check `test_output_chains.txt` to see if generated sentences are entity-rich.

## What to Check After Testing

### 1. Entity Density

Run the diagnostic tool and look for:
```
实体密度 (Entity Density): Should be >1.0 per sentence (was ~0.5)
```

### 2. Refusal Rate

```
拒答率 (Refusal Rate): Should be <15% (was 31.2% for musique, 27.2% for 2wiki)
```

### 3. Example Chains

Check `evaluate/outputs/stage2_predictions/musique_predictions_chains.txt` and verify:
- ✅ Sentences contain specific entities
- ✅ No generic statements like "We need to find more information"
- ✅ Graceful "I don't know" after sufficient attempts (not premature refusal)

### 4. EM Score

```bash
# Stage 3 will output EM scores
# Target: musique EM should improve from 0.124 to 0.17+ (5-point improvement)
```

## Metrics Comparison Table

Create a before/after comparison:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Refusal Rate (musique) | 31.2% | ??? | <15% |
| Refusal Rate (2wiki) | 27.2% | ??? | <15% |
| Avg Entities/Sentence | 0.5 | ??? | >1.5 |
| Entity-sparse Sentences | 30%+ | ??? | <10% |
| EM Score (musique) | 0.124 | ??? | 0.17+ |
| EM Score (2wiki) | ~0.12 | ??? | 0.17+ |

## Rollback Instructions

If results are worse, revert the changes:

```bash
cd D:\code\graduate\graduateRAG
git checkout llm_server/serve_llama_autobatch.py
```

Then restart the LLM server to load the old prompt.

## Next Steps Based on Results

### If Success (Refusal rate drops by 50%+, EM improves by 3+ points):
- ✅ Keep the changes
- ✅ Run on full datasets
- ✅ Document the improvement in thesis

### If Partial Success (Refusal rate drops 20-40%, EM improves by 1-2 points):
- Consider Phase 2: Query enhancement at retrieval layer
- See `RETRIEVAL_IMPROVEMENT_PLAN.md` Phase 2

### If No Improvement:
- Check if LLM server restarted with new prompt (check logs)
- Verify entity extraction is working (add debug logging)
- Consider if 8B model is too small to follow complex prompts
- Consider Phase 2: Query enhancement at retrieval layer

## Quick Validation Test

Before full evaluation, test on one question manually:

```python
# In Python interpreter
import requests

# Example question
prompt = """You are a multi-step reasoning agent...
Q: When was Neville A. Stanton's employer founded?
A: """

response = requests.post('http://localhost:8000/generate',
                        json={'prompt': prompt, 'max_tokens': 50})
print(response.json()['text'])

# Check if output contains entities like "University of Southampton"
# NOT generic like "We need to find information about his employer"
```

Expected output: `"Neville A. Stanton works at University of Southampton."`

Not: `"We need to find out where he works first."`
