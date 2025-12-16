# Single-hop dense (HNSW) retrieval config for Llama 3-8B
# Action: S-Dense - Single retrieval round with dense vectors

local retrieval_corpus_name = 'wiki';
local hnsw_retrieval_count = 10;

{
    "start_state": "generate_titles",
    "end_state": "[EOQ]",
    "models": {
        "generate_titles": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "generate_main_question",
            "retrieval_type": "hnsw",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": hnsw_retrieval_count,
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
            "prompt_file": "prompts/squad/gold_with_2_distractors_context_direct_qa_flan_t5.txt",
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
