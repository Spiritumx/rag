# RAG System Prompt Templates Documentation

This document provides a comprehensive overview of all prompt templates used across different datasets and retrieval strategies in the RAG evaluation pipeline.

## Table of Contents
1. [Overview](#overview)
2. [Strategy Z: Zero Retrieval](#strategy-z-zero-retrieval)
3. [Strategy S: Single-Hop Retrieval](#strategy-s-single-hop-retrieval)
4. [Strategy M: Multi-Hop Retrieval](#strategy-m-multi-hop-retrieval)
5. [Dataset-Specific Variations](#dataset-specific-variations)

---

## Overview

The system uses three main retrieval strategies:
- **Z (Zero Retrieval)**: Direct LLM generation without retrieval
- **S (Single-Hop)**: Single retrieval round with context (S-Sparse, S-Dense, S-Hybrid)
- **M (Multi-Hop)**: Multi-hop reasoning with MI-RA-ToT (Mutual Information Retrieval-Augmented Tree-of-Thought)

Each strategy uses dataset-specific prompts tailored to the characteristics of the evaluation dataset.

---

## Strategy Z: Zero Retrieval

### Purpose
Direct question answering without retrieval context, testing the LLM's parametric knowledge.

### File Location
`prompts/{dataset_name}/no_context_direct_qa_flan_t5.txt`

### Template Variants

#### SQuAD (Simple Template)
```
Please answer the following question directly.

Question:
[QUESTION]

Answer:
```

#### All Other Datasets (HotpotQA, TriviaQA, NQ, MuSiQue, 2WikiMultihopQA)
Uses few-shot learning with 28-30 examples:
```
# METADATA: {"qid": "..."}
Q: Answer the following question.
[Example Question]
A: [Example Answer]

[... 27 more examples ...]

# METADATA: {"qid": "..."}
Q: Answer the following question.
[Your Question]
A:
```

**Key Difference**: SQuAD uses zero-shot prompting, while multi-hop datasets use few-shot prompting to improve reasoning.

---

## Strategy S: Single-Hop Retrieval

### Purpose
Answer questions using a single retrieval round with 10 Wikipedia passages.

### File Location
`prompts/{dataset_name}/gold_with_2_distractors_context_direct_qa_flan_t5.txt`

### Common Template Structure
```
Please read the following context passages:

---------------------
Wikipedia Title: [Title 1]
[Content 1]

Wikipedia Title: [Title 2]
[Content 2]

... (10 passages total) ...
---------------------

Based strictly on the context above, answer the following question.

**IMPORTANT**: {DATASET_SPECIFIC_INSTRUCTION}

Question: [QUESTION]

Answer:
```

### Dataset-Specific Instructions

| Dataset | Answer Format Instruction |
|---------|--------------------------|
| **SQuAD** | Copy the exact span from the context. Do NOT paraphrase or summarize. |
| **HotpotQA** | Provide a single entity or short phrase as the answer. |
| **TriviaQA** | Provide only a brief entity or noun phrase (e.g., "Paris", "Albert Einstein"). Do NOT use complete sentences. |
| **NQ** | Provide a precise, short answer (entity or phrase) that directly addresses the question. Avoid over-generalization. |
| **MuSiQue** | Provide a precise entity or short text answer. Consider ALL relevant passages, not just a single piece of evidence. |
| **2WikiMultihopQA** | Provide a standardized entity name (e.g., use official names for people, places, organizations). |

**Rationale**:
- **SQuAD**: Extractive QA dataset → requires exact span extraction
- **HotpotQA**: Multi-hop dataset → may require synthesis but prefers concise entity answers
- **TriviaQA**: Factoid questions → entity-focused answers
- **NQ**: Open-domain questions → precise but allows slight flexibility
- **MuSiQue**: Complex multi-hop → emphasizes evidence integration
- **2WikiMultihopQA**: Wikipedia-based → prefers standardized entity names

---

## Strategy M: Multi-Hop Retrieval

### Purpose
Complex questions requiring iterative reasoning and multiple retrieval hops using the MI-RA-ToT framework.

### Architecture
Multi-hop strategy uses a two-stage approach:
1. **Stage 1**: Iterative reasoning with chain-of-thought (CoT)
2. **Stage 2**: Final answer synthesis

---

### Stage 1: Iterative Reasoning CoT

#### File Location
`prompts/{dataset_name}/stage1_iterative_reasoning_cot.txt`

#### Common Template
```
You are an expert research agent.
Follow the LOGICAL STRATEGY to find the answer step-by-step.

*** LOGICAL STRATEGY ***
{logical_plan}

*** RULES ***
1. {DATASET_SPECIFIC_RULE_1}
2. DO NOT repeat "Past Actions".
3. OUTPUT ONLY KEYWORDS (e.g., "Director Titanic", not "Who is the director").
4. If you have the final answer in "Current Info", "Action: Answer".

*** EXAMPLE ***
Question: "Who is the director of the film Titanic?"
Plan: Director(Titanic) -> [Person]
Hop 1:
Thought: I need to find the director of Titanic as per the plan.
Action: Search [Titanic film director]

*** YOUR TURN ***
Question: {query}

Past Actions:
{history_str}

Current Info:
{current_context_snippet}

Format:
Thought: <reasoning>
Action: <Search [Keywords] OR Answer>
```

#### Dataset-Specific Rule 1 Variations

| Dataset | Rule 1 Instruction |
|---------|-------------------|
| **SQuAD** | Execute the next step in the Logical Strategy. |
| **HotpotQA** | Execute the next step in the Logical Strategy. **Complete ALL steps before answering.** |
| **TriviaQA** | Execute the next step in the Logical Strategy. |
| **NQ** | Execute the next step in the Logical Strategy. |
| **MuSiQue** | Execute the next step in the Logical Strategy. **Gather sufficient evidence from multiple documents.** |
| **2WikiMultihopQA** | Execute the next step in the Logical Strategy. **Follow the correct reasoning order.** |

**Rationale**:
- **HotpotQA**: Emphasizes completing all reasoning steps (bridge questions)
- **MuSiQue**: Emphasizes evidence gathering (requires 2-4 hops)
- **2WikiMultihopQA**: Emphasizes reasoning order (compositional questions)

---

### Stage 2: Final Answer Synthesis

#### File Location
`prompts/{dataset_name}/stage2_final_answer_direct.txt`

#### Common Template
```
Task: Answer the complex question based on the Investigation Log and Retrieved Documents.

*** LOGICAL BLUEPRINT ***
{logical_plan}

*** INVESTIGATION LOG ***
{investigation_log}

*** RETRIEVED DOCUMENTS ***
{context_str}

*** INSTRUCTIONS ***
1. {DATASET_SPECIFIC_INSTRUCTION_1}
2. {DATASET_SPECIFIC_INSTRUCTION_2}

*** QUESTION ***
{query}

*** FORMAT ***
Thought: <analysis>
Answer: <concise answer>
```

#### Dataset-Specific Instructions

| Dataset | Instruction 1 | Instruction 2 |
|---------|--------------|--------------|
| **SQuAD** | Follow the Logical Blueprint to synthesize the answer. | Copy the exact span from the documents. Do NOT paraphrase or rewrite. |
| **HotpotQA** | Follow the Logical Blueprint to synthesize the answer. **Complete ALL reasoning steps.** | Provide a single entity or short phrase as the answer. |
| **TriviaQA** | Follow the Logical Blueprint to synthesize the answer. | Provide only a brief entity or noun phrase. Do NOT use complete sentences. |
| **NQ** | Follow the Logical Blueprint to synthesize the answer. | Provide a precise, short answer that directly addresses the question. Avoid over-generalization. |
| **MuSiQue** | Follow the Logical Blueprint to synthesize the answer. **Integrate evidence from ALL retrieved documents.** | Provide a precise entity or short text answer. |
| **2WikiMultihopQA** | Follow the Logical Blueprint to synthesize the answer. **Ensure correct reasoning order.** | Provide a standardized entity name (use official names for people, places, organizations). |

**Rationale**:
- **SQuAD**: Extractive dataset → exact span required
- **HotpotQA**: Multi-step questions → ensure all steps completed
- **MuSiQue**: Requires evidence from multiple documents → emphasize integration
- **2WikiMultihopQA**: Complex compositional questions → emphasize reasoning order

---

## Dataset-Specific Variations

### Summary Table

| Dataset | Question Type | Zero-Shot/Few-Shot | Answer Style | Multi-Hop Emphasis |
|---------|--------------|---------------------|--------------|-------------------|
| **SQuAD** | Extractive single-hop | Zero-shot | Exact span extraction | - |
| **HotpotQA** | Bridge multi-hop | Few-shot (28 examples) | Entity/short phrase | Complete all steps |
| **TriviaQA** | Factoid | Few-shot (28 examples) | Entity/noun phrase | Standard iteration |
| **NQ** | Open-domain | Few-shot (28 examples) | Precise short answer | Standard iteration |
| **MuSiQue** | Compositional multi-hop (2-4 hops) | Few-shot (20 examples) | Entity with evidence integration | Gather all evidence |
| **2WikiMultihopQA** | Wikipedia compositional | Few-shot (20 examples) | Standardized entity name | Correct reasoning order |

### Design Principles

1. **Answer Format Alignment**
   - Extractive datasets (SQuAD) → Exact span matching
   - Entity-focused datasets (TriviaQA, 2WikiMultihopQA) → Entity names
   - Open-domain datasets (NQ, HotpotQA) → Flexible short answers
   - Multi-evidence datasets (MuSiQue) → Integrated answers

2. **Reasoning Guidance**
   - Bridge questions (HotpotQA) → Complete all steps
   - Compositional questions (MuSiQue, 2WikiMultihopQA) → Evidence integration + reasoning order
   - Simple questions (SQuAD, TriviaQA, NQ) → Standard iteration

3. **Few-Shot Strategy**
   - Simple extractive tasks (SQuAD) → Zero-shot sufficient
   - Complex reasoning tasks (multi-hop datasets) → Few-shot with 20-28 examples

---

## File Mapping Reference

### Configuration Files
- **Baseline**: `evaluate/configs/llama_configs/{config_name}.jsonnet`
- **Innovation V2**: `innovation_experiments/evaluate_v2/configs/llama_configs_v2/{config_name}.jsonnet`

### Prompt Selection Logic
Prompts are selected dynamically using the `DATASET_NAME` environment variable:

```jsonnet
local dataset_name = std.extVar("DATASET_NAME");
"prompt_file": "prompts/" + dataset_name + "/{prompt_template}.txt"
```

### Strategy to Prompt Mapping

| Strategy | Config File | Prompt Template |
|----------|------------|----------------|
| Z | `zero_retrieval.jsonnet` | `no_context_direct_qa_flan_t5.txt` |
| S-Sparse | `single_splade.jsonnet` | `gold_with_2_distractors_context_direct_qa_flan_t5.txt` |
| S-Dense | `single_dense.jsonnet` | `gold_with_2_distractors_context_direct_qa_flan_t5.txt` |
| S-Hybrid | `single_hybrid.jsonnet` | `gold_with_2_distractors_context_direct_qa_flan_t5.txt` |
| M | `multi_hop.jsonnet` | Stage 1: `stage1_iterative_reasoning_cot.txt`<br>Stage 2: `stage2_final_answer_direct.txt` |

---

## Usage Example

### Running with Different Datasets
```bash
# SQuAD
DATASET_NAME=squad CORPUS_NAME=squad python evaluate.py

# HotpotQA
DATASET_NAME=hotpotqa CORPUS_NAME=wiki python evaluate.py

# MuSiQue
DATASET_NAME=musique CORPUS_NAME=wiki python evaluate.py
```

### Impact on Pipeline
1. **Stage 1 (Classification)**: Dataset-agnostic, uses same router model
2. **Stage 2 (Generation)**:
   - Loads dataset-specific prompts based on `DATASET_NAME`
   - Selects corpus based on `CORPUS_NAME`
3. **Stage 3 (Evaluation)**: Dataset-specific metrics (EM, F1, etc.)

---

## Maintenance Notes

### Adding a New Dataset
1. Create prompt directory: `prompts/{new_dataset}/`
2. Create three prompt files:
   - `no_context_direct_qa_flan_t5.txt` (Z strategy)
   - `gold_with_2_distractors_context_direct_qa_flan_t5.txt` (S strategies)
   - `stage1_iterative_reasoning_cot.txt` (M strategy, stage 1)
   - `stage2_final_answer_direct.txt` (M strategy, stage 2)
3. Add dataset to config: `config_v2.yaml` datasets list
4. No code changes needed (dynamic prompt loading)

### Modifying Prompts
- **DO NOT** change placeholder variables: `[QUESTION]`, `{query}`, `{logical_plan}`, etc.
- Test on sample questions before deployment
- Document changes in this file

---

## References

- **IRCoT Framework**: Multi-hop reasoning approach used in M strategy
- **Llama 3-8B**: Base LLM model (8000 token context limit)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (for all single/multi-hop strategies)

---

*Last Updated: 2026-01-11*
*Version: 2.0 (Innovation Experiments)*
