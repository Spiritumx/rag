"""
FastAPI LLM Server with Batch Inference Support for Llama models.

Features:
- Single and batch inference endpoints
- Dynamic batching for better GPU utilization
- Compatible with existing llm_client_generator.py

Usage:
    export MODEL_PATH="/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
    export LLM_PORT=8000
    pixi run python llm_server/serve_llama.py
"""

import os
import time
from typing import List, Optional
from functools import lru_cache

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", os.environ.get("LLM_MODEL_PATH", "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"))
MODEL_NAME = os.path.basename(MODEL_PATH)  # Extract model name from path

# Global model and tokenizer
model = None
tokenizer = None


@lru_cache(maxsize=None)
def get_model_and_tokenizer():
    """Load model and tokenizer (cached)."""
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return model, tokenizer

    print(f"Loading model from {MODEL_PATH}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL: Use left padding for batched generation
    # Right padding causes misalignment between prompt and generated tokens
    tokenizer.padding_side = 'left'

    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True
    )

    model.eval()  # Set to evaluation mode

    print(f"✓ Model loaded: {MODEL_NAME}")
    print(f"✓ Device: {next(model.parameters()).device}")
    print(f"✓ Dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


class GenerateRequest(BaseModel):
    """Single generation request."""
    prompt: str
    max_input: Optional[int] = None
    max_length: int = 200
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    num_return_sequences: int = 1
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    keep_prompt: bool = False


class BatchGenerateRequest(BaseModel):
    """Batch generation request."""
    prompts: List[str]
    max_input: Optional[int] = None
    max_length: int = 200
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    keep_prompt: bool = False


@app.get("/")
async def index():
    """Health check endpoint."""
    return {
        "message": f"Llama Batch Server running",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH
    }


@app.get("/generate")
@app.get("/generate/")
async def generate_get(
    prompt: str,
    max_input: Optional[int] = None,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: Optional[float] = None,
    length_penalty: Optional[float] = None,
    keep_prompt: bool = False,
):
    """Generate endpoint (GET method - compatible with llm_client_generator.py)."""
    return await generate_single(
        prompt=prompt,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt
    )


@app.post("/generate")
async def generate_post(request: GenerateRequest):
    """Generate endpoint (POST method)."""
    return await generate_single(
        prompt=request.prompt,
        max_input=request.max_input,
        max_length=request.max_length,
        min_length=request.min_length,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        num_return_sequences=request.num_return_sequences,
        repetition_penalty=request.repetition_penalty,
        length_penalty=request.length_penalty,
        keep_prompt=request.keep_prompt
    )


async def generate_single(
    prompt: str,
    max_input: Optional[int] = None,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: Optional[float] = None,
    length_penalty: Optional[float] = None,
    keep_prompt: bool = False,
):
    """
    Generate text for a single prompt.
    Compatible with llm_client_generator.py interface.
    """
    start_time = time.time()

    model, tokenizer = get_model_and_tokenizer()

    # Tokenize with truncation if max_input specified
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True if max_input else False,
        max_length=max_input,
        padding=False
    ).to(model.device)

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": max_length,
        "min_new_tokens": min_length,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 1.0,
        "top_k": top_k if top_k > 0 else None,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if length_penalty is not None:
        gen_kwargs["length_penalty"] = length_penalty

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode - get prompt token length for accurate removal
    prompt_token_length = inputs.input_ids.shape[1]

    if not keep_prompt:
        # Remove prompt tokens before decoding (more accurate than string slicing)
        outputs_without_prompt = outputs[:, prompt_token_length:]
        generated_texts = tokenizer.batch_decode(outputs_without_prompt, skip_special_tokens=True)
    else:
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Clean up (strip whitespace)
    generated_texts = [text.strip() for text in generated_texts]

    # Count tokens
    generated_num_tokens = [len(tokenizer.encode(text)) for text in generated_texts]

    end_time = time.time()

    return {
        "generated_texts": generated_texts,
        "generated_num_tokens": generated_num_tokens,
        "run_time_in_seconds": end_time - start_time,
        "model_name": MODEL_NAME,
    }


@app.post("/generate_batch")
async def generate_batch(request: BatchGenerateRequest):
    """
    Generate text for multiple prompts in a single batch.
    Much more efficient than multiple single requests.

    Example:
        {
            "prompts": ["prompt1", "prompt2", "prompt3"],
            "max_length": 200,
            "temperature": 1.0
        }
    """
    start_time = time.time()

    model, tokenizer = get_model_and_tokenizer()

    # Batch tokenization
    inputs = tokenizer(
        request.prompts,
        return_tensors="pt",
        truncation=True if request.max_input else False,
        max_length=request.max_input,
        padding=True,  # Pad to same length for batching
        return_attention_mask=True
    ).to(model.device)

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": request.max_length,
        "min_new_tokens": request.min_length,
        "do_sample": request.do_sample,
        "temperature": request.temperature if request.do_sample else 1.0,
        "top_k": request.top_k if request.top_k > 0 else None,
        "top_p": request.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": inputs.attention_mask,
    }

    if request.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = request.repetition_penalty
    if request.length_penalty is not None:
        gen_kwargs["length_penalty"] = request.length_penalty

    # Batch generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **gen_kwargs
        )

    # Decode all outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Remove prompts if needed
    if not request.keep_prompt:
        generated_texts = [
            text[len(prompt):].strip()
            for text, prompt in zip(generated_texts, request.prompts)
        ]

    # Count tokens for each output
    generated_num_tokens = [len(tokenizer.encode(text)) for text in generated_texts]

    end_time = time.time()

    return {
        "generated_texts": generated_texts,
        "generated_num_tokens": generated_num_tokens,
        "run_time_in_seconds": end_time - start_time,
        "batch_size": len(request.prompts),
        "model_name": MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    print(f"""
╔════════════════════════════════════════════════════════════╗
║           Llama Batch Inference Server                     ║
╚════════════════════════════════════════════════════════════╝

Model: {MODEL_PATH}
Port: {os.environ.get("LLM_PORT", "8000")}

Endpoints:
  GET  /generate/        - Single inference (query params)
  POST /generate         - Single inference (JSON body)
  POST /generate_batch   - Batch inference (JSON array)

Loading model...
""")

    # Preload model
    get_model_and_tokenizer()

    print("\n✓ Server ready!\n")

    port = int(os.environ.get("LLM_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
