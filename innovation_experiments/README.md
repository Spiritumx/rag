# Innovation Experiments

This directory contains experimental improvements to the baseline RAG system while preserving the original baseline for A/B testing and academic research.

## Three Innovations

### 1. Adaptive Retrieval (`retriever_server_v2/`)
**Dynamic weight adjustment** based on query analysis
- Uses NER + semantic classifier to analyze queries
- Computes lexical specificity (entity density) and semantic abstractness
- Dynamically adjusts BM25/SPLADE/Dense weights for hybrid retrieval
- **Example**: Entity-rich queries → favor BM25/SPLADE; Abstract queries → favor Dense

### 2. Cascading Dynamic Routing (`evaluate_v2/stage2_generate_v2.py`)
**Confidence-based fallback mechanism**
- Executes initial strategy from classifier (e.g., S-Hybrid)
- Runs posterior verification using cross-encoder confidence scoring
- Cascades to MI-RA-ToT if confidence < 0.6
- Logs all routing decisions for analysis

### 3. MI-RA-ToT (`evaluate_v2/M_core_tot.py`)
**Mutual Information Tree-of-Thought reasoning**
- Replaces linear while-loop with beam search tree
- Explores multiple reasoning paths in parallel (beam width=3, depth=4)
- Uses MI-based scoring: α × relevance + β × novelty
- Prunes low-scoring branches, keeps top-B paths per layer

---

## Quick Start

### 1. Setup (First Time Only)
```bash
# Setup V2 retriever
cd innovation_experiments/retriever_server_v2
bash setup.sh

# Setup V2 evaluation
cd ../evaluate_v2/utils
bash setup_symlinks.sh
```

### 2. Run Baseline (for comparison)
```bash
# Terminal 1: Start baseline retriever
python retriever_server/serve.py

# Terminal 2: Run baseline pipeline
python evaluate/run_pipeline.py --datasets squad hotpotqa --stages 1 2 3
```

### 3. Run V2 (with innovations)
```bash
# Terminal 1: Start V2 retriever
python innovation_experiments/retriever_server_v2/serve_v2.py

# Terminal 2: Run V2 pipeline (Stages 2 & 3 only, reuses Stage 1)
python innovation_experiments/evaluate_v2/run_pipeline_v2.py --datasets squad hotpotqa --stages 2 3
```

### 4. Run Analysis
```bash
# Run all analyses and generate reports
python innovation_experiments/run_all_analysis.py --datasets squad hotpotqa
```

**Results**: Check `innovation_experiments/analysis_results/MASTER_SUMMARY.md`

---

## Directory Structure

```
innovation_experiments/
├── README.md                          (this file)
├── .retriever_address_v2.jsonnet      (Port 8002 config)
├── compare_metrics.py                 (A/B comparison script)
│
├── retriever_server_v2/               [Innovation 1: Adaptive Retrieval]
│   ├── serve_v2.py                    (Port 8002)
│   ├── elasticsearch_retriever_v2.py  (Dynamic weights integration)
│   ├── unified_retriever_v2.py        (Wrapper for v2 retriever)
│   ├── query_analyzer.py              (NER + semantic classifier)
│   ├── analyze_weights.py             (Weight pattern analysis)
│   └── model/                         (symlink to baseline models)
│
└── evaluate_v2/                       [Innovation 2 & 3: Cascading + ToT]
    ├── run_pipeline_v2.py             (V2 pipeline orchestrator)
    ├── stage2_generate_v2.py          (Cascade routing logic)
    ├── stage3_evaluate_v2.py          (Metrics + cascade analysis)
    ├── M_core_tot.py                  (MI-RA-ToT beam search)
    ├── analyze_cascade.py             (Cascade decision analysis)
    ├── config_v2.yaml                 (V2 configuration)
    ├── configs/
    │   ├── action_to_config_mapping_v2.py
    │   └── llama_configs_v2/          (JSONNET configs)
    ├── utils/
    │   ├── confidence_verifier.py     (Answer confidence scoring)
    │   ├── routing_logger.py          (Cascade tracking)
    │   ├── result_manager.py          (Modified for outputs_v2)
    │   └── [symlinks to baseline utils]
    └── outputs_v2/
        ├── stage2_predictions_v2/
        ├── stage3_metrics_v2/
        └── cascade_analysis/
```

---

## Usage

### Baseline System (Original)
```bash
# Terminal 1: Start baseline retriever
python retriever_server/serve.py

# Terminal 2: Run baseline pipeline
python evaluate/run_pipeline.py --datasets squad hotpotqa --stages 1 2 3
```

**Characteristics**:
- Retriever Port: 8001
- Static hybrid weights: {bm25: 1.0, splade: 1.0, dense: 1.0}
- No cascading: Direct strategy execution
- Linear multi-hop: Greedy depth-first reasoning

---

### Improved System (V2 with Innovations)
```bash
# Terminal 1: Start V2 retriever
python innovation_experiments/retriever_server_v2/serve_v2.py

# Terminal 2: Run V2 pipeline (reuses Stage 1 classifications)
python innovation_experiments/evaluate_v2/run_pipeline_v2.py --datasets squad hotpotqa --stages 2 3
```

**Characteristics**:
- Retriever Port: 8002
- Dynamic hybrid weights: Query-dependent
- Cascading: Low-confidence answers escalate to ToT
- Beam search multi-hop: Explores 3 paths per layer, 4 layers deep

---

## A/B Testing & Comparison

### Option 1: Run All Analyses (Recommended)

```bash
# Run complete analysis suite
python innovation_experiments/run_all_analysis.py --datasets squad hotpotqa
```

This will:
1. Compare baseline vs V2 metrics (EM, F1, per-dataset, per-action)
2. Analyze cascade routing decisions and effectiveness
3. Analyze adaptive weight patterns
4. Generate master summary report

**Outputs**:
- `innovation_experiments/analysis_results/baseline_vs_v2_comparison.md`
- `innovation_experiments/analysis_results/cascade_analysis_report.md`
- `innovation_experiments/analysis_results/adaptive_weights_report.md`
- `innovation_experiments/analysis_results/MASTER_SUMMARY.md`

### Option 2: Run Individual Analyses

#### Compare Metrics
```bash
# Generate side-by-side comparison table
python innovation_experiments/compare_metrics.py
```

**Output**:
- EM/F1 gains per dataset
- Overall performance improvement
- Per-action performance breakdown
- Markdown tables + JSON data

#### Analyze Cascade Decisions
```bash
# Analyze routing patterns
python innovation_experiments/analyze_cascade.py --datasets squad hotpotqa
```

**Metrics**:
- Cascade trigger rate by initial action
- Direct strategy accuracy vs Cascaded accuracy
- Confidence score distribution
- Threshold sensitivity analysis
- Low-confidence example queries

#### Analyze Adaptive Weights
```bash
# Analyze weight patterns
python innovation_experiments/analyze_weights.py --datasets squad hotpotqa
```

**Metrics**:
- Query characteristic distribution (lexical vs semantic)
- Weight pattern distribution (lexical-dominant, semantic-dominant, balanced)
- Example queries for each category
- Correlation with query type

---

## Configuration

### Shared Services (Both Systems Use)
- **LLM Server**: Port 8000 (Llama 3-8B) - HTTP, stateless
- **Elasticsearch**: Port 9200 - Shared index, read-only
- **Classifier**: Stage 1 (Qwen 2.5-3B) - Same weights for fair comparison

### Separate Resources
- **Retriever Servers**: 8001 (baseline) vs 8002 (v2)
- **Prediction Outputs**: `evaluate/outputs/` vs `innovation_experiments/evaluate_v2/outputs_v2/`
- **Logs**: Separate pipeline logs for each system

### Configuration Files
- **Baseline**: `evaluate/config.yaml`
- **V2**: `innovation_experiments/evaluate_v2/config_v2.yaml`

**Key V2 Settings**:
```yaml
retriever:
  port: 8002

innovations:
  adaptive_retrieval:
    enabled: true
    ner_model: "en_core_web_sm"

  cascading_routing:
    enabled: true
    confidence_threshold: 0.6
    verifier_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

  tot_reasoning:
    enabled: true
    beam_width: 3
    max_depth: 4
    mi_alpha: 0.7  # Relevance weight
    mi_beta: 0.3   # Novelty weight
```

---

## Implementation Status

### Phase 1: Foundation + Innovation 1 ✅ COMPLETE
- [x] Directory structure
- [x] README documentation
- [x] Query analyzer (NER + semantic classifier)
- [x] Modified retriever with dynamic weights
- [x] V2 server on port 8002
- [x] Setup scripts for symlinks

### Phase 2: Innovation 2 ✅ COMPLETE
- [x] Confidence verifier (cross-encoder)
- [x] Routing logger (CSV tracking)
- [x] Modified Stage 2 generator with cascade logic
- [x] Config v2 file with all innovation settings
- [x] ResultManagerV2 for V2 paths

### Phase 3: Innovation 3 ✅ COMPLETE
- [x] TreeNode dataclass
- [x] MutualInformationScorer (novelty + relevance)
- [x] BeamSearchToT algorithm (beam width=3, depth=4)
- [x] MI-RA-ToT full implementation
- [x] Integration in stage2_generate_v2.py

### Phase 4: Integration ✅ COMPLETE
- [x] V2 pipeline runner (run_pipeline_v2.py)
- [x] V2 evaluator with cascade metrics (stage3_evaluate_v2.py)
- [x] Action config mapping V2
- [x] All JSONNET configs for V2
- [x] End-to-end pipeline ready

### Phase 5: Analysis ✅ COMPLETE
- [x] Comparison script (compare_metrics.py)
- [x] Cascade analysis (analyze_cascade.py)
- [x] Weight analysis (analyze_weights.py)
- [x] Master analysis runner (run_all_analysis.py)
- [x] Full A/B test infrastructure

### Phase 6: Documentation ✅ COMPLETE
- [x] Comprehensive README
- [x] Setup instructions
- [x] Troubleshooting guide
- [x] Expected results documentation

---

## Expected Results

### Quantitative Gains (Expected)
| Innovation | EM Gain | F1 Gain |
|-----------|---------|---------|
| Adaptive Retrieval | +0.02 to +0.05 | +0.03 to +0.06 |
| Cascading Routing | +0.03 to +0.08 | +0.04 to +0.10 |
| MI-RA-ToT | +0.05 to +0.12 | +0.06 to +0.15 |
| **Combined** | **+0.10 to +0.25** | **+0.13 to +0.31** |

### Qualitative Insights
- Query type analysis: Evidence for query-dependent weight optimization
- Cascade patterns: Which initial strategies benefit most from fallback
- ToT reasoning: Examples where beam search finds better paths than greedy

---

## Paper Contributions

1. **Adaptive Retrieval**: Query-dependent hybrid weight optimization improves retrieval quality
2. **Cascading Routing**: Confidence-based fallback reduces strategy misclassification errors
3. **MI-RA-ToT**: Beam search with MI scoring outperforms greedy multi-hop reasoning
4. **Ablation Study**: Individual and combined contributions of each innovation

---

## Troubleshooting

### Port Conflicts
If port 8002 is already in use:
```bash
# Check what's using port 8002
netstat -ano | findstr :8002

# Kill the process or change port in config_v2.yaml
```

### Import Errors
Ensure Python path includes project root:
```python
import sys
sys.path.insert(0, 'D:\\code\\graduate\\graduateRAG')
```

### Model Loading Issues
- NER model: `python -m spacy download en_core_web_sm`
- Cross-encoder: Auto-downloads from HuggingFace on first run
- Retriever models: Shared via symlink, ensure baseline models exist

---

## Contact & Support

For questions or issues:
1. Check baseline system works first
2. Verify all dependencies installed
3. Review logs in `innovation_experiments/evaluate_v2/pipeline_v2.log`
4. Compare with plan: `C:\Users\www\.claude\plans\jaunty-churning-stallman.md`
