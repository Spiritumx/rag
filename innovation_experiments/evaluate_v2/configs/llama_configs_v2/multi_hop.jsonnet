# [V2 Config] Multi-hop reasoning config for Llama 3-8B
# Action: M - MI-RA-ToT beam search reasoning (Innovation 3)
# Process: Beam search with MI-based path scoring → Best path → Final answer
# NOTE: This config is used for fallback/compatibility. ToT logic is in stage2_generate_v2.py

local retrieval_corpus_name = 'wiki';  # Use wiki for all datasets
local retrieval_count = 10;  # Final top-K after reranking per iteration
local max_buffer_count = 20;  # Number of docs to retrieve before reranking

{
    "start_state": "step_by_step_hybrid_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_hybrid_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "hybrid",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": retrieval_count,
            "max_buffer_count": max_buffer_count,
            "global_max_num_paras": 40,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "return_pids": false,
            "cumulate_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": "prompts/hotpotqa/gold_with_2_distractors_context_cot_qa_codex.txt",
            "question_prefix": "Answer the following question by reasoning step-by-step.\n",
            "prompt_reader_args": {
                "shuffle": false,
                "model_length_limit": 1000000,
                "tokenizer_model_name": "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct",
            },
            "generation_type": "sentences",
            "reset_queries_as_sentences": false,
            "add_context": true,
            "shuffle_paras": false,
            "terminal_return_type": null,
            "disable_exit": true,
            "max_num_sentences": 20,
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "Meta-Llama-3-8B-Instruct",
            "model_tokens_limit": 8000,
            "max_length": 200,
        },
        "step_by_step_exit_controller": {
            "name": "step_by_step_exit_controller",
            "next_model": "step_by_step_hybrid_retriever",
            // Improved regex: handles "answer is:", "A:", or direct answer formats
            // Captures answer until period or newline, more robust than before
            "answer_extractor_regex": "(?:.*?(?:answer is:?|A:)\\s*|^)([^.\\n]+?)(?:\\.|\\n|$)",
            "answer_extractor_remove_last_fullstop": true,
            "terminal_state_next_model": "generate_main_question",
            "terminal_return_type": "pids",
            "max_num_sentences": 20,
            "global_max_num_paras": 40,
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
            "next_model": "extract_answer",
            "prompt_file": "prompts/hotpotqa/gold_with_2_distractors_context_direct_qa_codex.txt",
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
        "extract_answer": {
            "name": "answer_extractor",
            "query_source": "last_answer",
            // Improved regex: handles "answer is:", "A:", or direct answer formats
            // Captures answer until period or newline, more robust than before
            "regex": "(?:.*?(?:answer is:?|A:)\\s*|^)([^.\\n]+?)(?:\\.|\\n|$)",
            "match_all_on_failure": true,
            "remove_last_fullstop": true,
        }
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": false,
    },
    "prediction_type": "answer"
}
