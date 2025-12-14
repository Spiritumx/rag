# Zero retrieval config for Llama 3-8B
# Action: Z - Direct LLM generation without retrieval

local add_pinned_paras = false;

{
    "start_state": "generate_question",
    "end_state": "[EOQ]",
    "models": {
        "generate_question": {
            "name": "copy_question",
            "next_model": "generate_answer",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "generate_answer": {
            "name": "llmqa",
            "next_model": null,
            "prompt_file": "prompts/squad/no_context_direct_qa_flan_t5.txt",
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
            "add_context": false,
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
