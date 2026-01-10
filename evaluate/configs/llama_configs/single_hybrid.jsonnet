# Single-hop hybrid retrieval config for Llama 3-8B
# Action: S-Hybrid - Hybrid retrieval (BM25 + SPLADE + HNSW) with Reranker
# Process: Each method retrieves 20 docs → merge → Reranker → Top-10

local dataset_name = std.extVar("DATASET_NAME");  # Dynamic dataset for prompt selection
local retrieval_corpus_name = std.extVar("CORPUS_NAME");  # Dynamic corpus based on dataset
local retrieval_count = 10;  # Final top-K after reranking
local max_buffer_count = 20;  # Number of docs to retrieve before reranking

{
    "start_state": "generate_titles",
    "end_state": "[EOQ]",
    "models": {
        "generate_titles": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "generate_main_question",
            "retrieval_type": "hybrid",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": retrieval_count,
            "max_buffer_count": max_buffer_count,
            "global_max_num_paras": 15,
            "query_source": "original_question",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "end_state": "[EOQ]",
        },
        "generate_main_question": {
            "name": "copy_question",
            "next_model": "answer_main_question",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "answer_main_question": {
            "name": "llmqa",
            "next_model": null,
            "prompt_file": "prompts/" + dataset_name + "/gold_with_2_distractors_context_direct_qa_flan_t5.txt",
            "question_prefix": "Answer the following question.\n",
            "prompt_reader_args": {
                "shuffle": false,
                "model_length_limit": 1000000,
                "tokenizer_model_name": "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct",
            },
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "Meta-Llama-3-8B-Instruct",
            "model_tokens_limit": 8000,
            "max_length": 200,
            "add_context": true,
        },
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": false,
    },
    "prediction_type": "answer"
}
