# Thesis TODO List

## Experiments to Run

### 1. Classifier Evaluation (Ch03)
- **Priority**: High
- **Description**: Compute per-class Precision/Recall/F1 for the Z/S/M three-class classifier
- **Data source**: `evaluate/outputs/stage1_classifications/` vs ground truth labels
- **Affects**: `chapter03.tex` Table `tab:classifier_metrics`, confusion matrix figure, and percentage values in error analysis paragraphs

### 2. Ch04 Ablation Experiments
- **Priority**: High
- **Description**: Run ablation experiments for 3 variants:
  - V2-full vs V2-without-ToT (tree-of-thought)
  - V2-full vs V2-without-Cascade (cascade routing)
  - V2-full vs V2-without-DynamicRetrieval (dynamic weight retrieval)
- **Script**: `innovation_experiments/run_ablation_experiments.py`
- **Affects**: `chapter04.tex` Table `tab:chapter4_ablation_results`, ToT effectiveness analysis, chapter summary

### 3. Latency Measurement
- **Priority**: Medium
- **Description**: Time baseline system vs V2 system end-to-end per query
  - Measure avg seconds/question for baseline system
  - Measure avg seconds/question for V2 system
  - Break down by: classification time, retrieval time, generation time, cascade overhead
- **Affects**: `chapter04.tex` latency paragraphs (sections 4.5.2 and 4.5.4)

### 4. Dynamic vs Static Retrieval Recall@5 Comparison
- **Priority**: Medium
- **Description**: Run retrieval with both static weight config and dynamic weight config, compare Recall@5 per dataset
  - Specifically need NQ Recall@5 improvement and HotpotQA Recall@5 improvement
- **Affects**: `chapter04.tex` section 4.5.1 dynamic retrieval analysis

### 5. Evidence Document Count Analysis
- **Priority**: Low
- **Description**: Parse `*_predictions_chains_v2.txt` files to count unique evidence documents per query
  - Compare MI-RA-ToT vs linear reasoning strategy
  - Report average increase in unique evidence documents
- **Affects**: `chapter04.tex` section 4.5.3 multi-hop reasoning depth analysis

---

## Plots to Generate

### Ch03 Plots

1. **Classifier Confusion Matrix Heatmap** (depends on Experiment #1)
   - 3-class (Z/S/M) confusion matrix as heatmap
   - Output: `thesis/imgs/chapter03/confusion_matrix.pdf`
   - Referenced as: `fig:confusion_matrix`

2. **Overall EM/F1 Bar Chart - Four Methods**
   - Bar chart comparing Z, S-Hybrid, Adaptive-RAG, Proposed across EM and F1
   - Data: Table `tab:overall_comparison`

3. **Per-Dataset EM Grouped Bar Chart - Four Methods**
   - Grouped bar chart: 6 datasets x 4 methods, EM metric
   - Data: Table `tab:per_dataset_em_comparison`

4. **Action Distribution Stacked Bar Chart**
   - Stacked bar chart showing Z/S-Sparse/S-Dense/S-Hybrid/M distribution per dataset
   - Data: Table `tab:action_distribution`

5. **Single-Strategy Ablation Bar Chart**
   - Bar chart comparing Z, S-Sparse, S-Dense, S-Hybrid, M across EM/F1
   - Data: Table `tab:ablation_single_strategy`

### Ch04 Plots

6. **Baseline vs V2 Per-Dataset EM/F1 Grouped Bar Chart**
   - Grouped bar chart: 6 datasets, baseline EM vs V2 EM (and/or F1)
   - Data: Table `tab:chapter4_overall_results`

7. **Cascade Trigger Rate Bar Chart**
   - Bar chart showing cascade trigger rate per dataset
   - Data: Table `tab:cascade_per_dataset`

8. **Dynamic Weight Distribution Visualization**
   - Show BM25/SPLADE/Dense weight distribution for entity-dense vs semantic-abstract queries
   - Example: NQ (BM25~0.52, SPLADE~0.32, Dense~0.16) vs HotpotQA (BM25~0.14, SPLADE~0.28, Dense~0.58)

9. **Ablation Experiment Grouped Bar Chart** (depends on Experiment #2)
   - Bar chart comparing full V2 vs 3 ablation variants across EM/F1
   - Data: Table `tab:chapter4_ablation_results` (to be created after experiment)

10. **Multi-hop vs Single-hop Performance Improvement Chart**
    - Grouped bar or paired bar showing delta-EM for multi-hop datasets vs single-hop datasets
    - Data from Table `tab:chapter4_overall_results`

---

## Remaining Placeholder Values

### chapter03.tex
- `tab:classifier_metrics`: All P/R/F1 values (depends on Experiment #1)
- Confusion matrix percentages: S->M misclassification rate, M->S misclassification rate (depends on Experiment #1)
- Classifier overall accuracy and macro-F1 in text (depends on Experiment #1)

### chapter04.tex
- ToT effectiveness: EM improvement over linear strategy (depends on Experiment #2)
- Evidence document count increase (depends on Experiment #5)
- `tab:chapter4_ablation_results`: Full ablation table (depends on Experiment #2)
- `tab:chapter4_hardset_results`: Hard dataset comparison table (depends on Experiment #2)
- `fig:chapter4_reasoning_path`: Reasoning path visualization (needs to be created)
- Latency values: baseline vs V2 seconds/question (depends on Experiment #3)
- Dynamic retrieval Recall@5 improvements (depends on Experiment #4)
