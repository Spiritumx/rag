"""
FastAPI LLM Server with Automatic Request Batching for Llama models.
Updated for General Purpose Agentic RAG.
"""

import os
import time
import asyncio
from typing import List, Optional, Dict, Any
from functools import lru_cache
from dataclasses import dataclass
from queue import Queue
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", os.environ.get("LLM_MODEL_PATH", "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"))
MODEL_NAME = os.path.basename(MODEL_PATH)

# Batch configuration
MAX_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", "50"))

# Global model and tokenizer
model = None
tokenizer = None

# Request queue and processing
request_queue = Queue()
response_dict: Dict[str, Any] = {}
response_events: Dict[str, threading.Event] = {}
request_id_counter = 0
request_id_lock = threading.Lock()

def apply_llama3_template(raw_content: str) -> str:
    """
    Standard Llama-3 Chat Template Wrapper.
    
    IMPORTANT: This version is 'strategy-agnostic'. 
    It trusts that the client (M_core.py) provides the specific instructions (Task, Context, Rules).
    It simply wraps the client's prompt in the official Llama-3 special tokens.
    """
    
    # Llama-3 Special Tokens
    bos = "<|begin_of_text|>"
    header_start = "<|start_header_id|>"
    header_end = "<|end_header_id|>"
    eot = "<|eot_id|>"

    # Generic System Prompt
    # 我们使用一个通用的、顺从的系统提示词，让模型专注于 User Prompt 中的指令
    system_prompt = "You are a helpful, intelligent AI assistant. Follow the user's instructions carefully and precisely."

    # Construct the formatted string
    formatted_prompt = (
        f"{bos}"
        f"{header_start}system{header_end}\n\n{system_prompt}{eot}"
        f"{header_start}user{header_end}\n\n{raw_content}{eot}"
        f"{header_start}assistant{header_end}\n\n"
    )

    return formatted_prompt

@dataclass
class BatchRequest:
    """Internal batch request representation."""
    request_id: str
    prompt: str
    max_input: Optional[int]
    max_length: int
    min_length: int
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    num_return_sequences: int
    repetition_penalty: Optional[float]
    length_penalty: Optional[float]
    keep_prompt: bool


def get_next_request_id():
    """Generate unique request ID."""
    global request_id_counter
    with request_id_lock:
        request_id_counter += 1
        return f"req_{request_id_counter}"


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

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # CRITICAL: Use left padding for batched generation
    tokenizer.padding_side = 'left'

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True
    )

    model.eval()

    print(f"✓ Model loaded: {MODEL_NAME}")
    print(f"✓ Device: {next(model.parameters()).device}")
    print(f"✓ Batch size: {MAX_BATCH_SIZE}, Timeout: {BATCH_TIMEOUT_MS}ms")

    return model, tokenizer


def batch_processing_worker():
    """
    Background worker that processes batched requests.
    """
    model, tokenizer = get_model_and_tokenizer()
    print("✓ Batch processing worker started")

    while True:
        # Collect batch
        batch = []
        start_collect = time.time()
        timeout_seconds = BATCH_TIMEOUT_MS / 1000.0

        try:
            first_req = request_queue.get(timeout=1.0)
            batch.append(first_req)
        except:
            continue

        while len(batch) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_collect
            remaining_timeout = max(0.001, timeout_seconds - elapsed)

            try:
                req = request_queue.get(timeout=remaining_timeout)
                batch.append(req)
            except:
                break

        if batch:
            process_batch(batch, model, tokenizer)


def process_batch(batch: List[BatchRequest], model, tokenizer):
    """Process a batch of requests."""
    if len(batch) == 1:
        process_single_request(batch[0], model, tokenizer)
        return

    try:
        # Group by generation parameters
        param_groups = {}
        for req in batch:
            key = (
                req.max_length, req.min_length, req.do_sample,
                req.temperature, req.top_k, req.top_p,
                req.repetition_penalty, req.length_penalty
            )
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(req)

        for group in param_groups.values():
            if len(group) == 1:
                process_single_request(group[0], model, tokenizer)
            else:
                process_batch_group(group, model, tokenizer)

    except Exception as e:
        print(f"Batch processing error: {e}, falling back to individual")
        for req in batch:
            process_single_request(req, model, tokenizer)

def process_batch_group(group: List[BatchRequest], model, tokenizer):
    """Process a group of requests with same parameters as a true batch."""
    start_time = time.time()

    # Apply template
    prompts = [apply_llama3_template(req.prompt) for req in group]
    sample_req = group[0]

    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True if sample_req.max_input else False,
        max_length=sample_req.max_input,
        padding=True,
        return_attention_mask=True
    ).to(model.device)

    input_seq_len = inputs.input_ids.shape[1]
    
    # Terminator tokens (Llama 3 specific)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    gen_kwargs = {
        "max_new_tokens": sample_req.max_length,
        "min_new_tokens": sample_req.min_length,
        "do_sample": sample_req.do_sample,
        "temperature": sample_req.temperature if sample_req.do_sample else 1.0,
        "top_k": sample_req.top_k if sample_req.top_k > 0 else None,
        "top_p": sample_req.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": terminators, # Use explicit terminators
        "attention_mask": inputs.attention_mask,
    }

    if sample_req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = sample_req.repetition_penalty
    if sample_req.length_penalty is not None:
        gen_kwargs["length_penalty"] = sample_req.length_penalty

    # Generate
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **gen_kwargs)

    # Decode
    new_tokens = outputs[:, input_seq_len:]
    generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Return results
    for i, req in enumerate(group):
        text = generated_texts[i].strip()
        if req.keep_prompt:
            text = req.prompt + text

        num_tokens = len(tokenizer.encode(text))
        result = {
            "generated_texts": [text],
            "generated_num_tokens": [num_tokens],
            "run_time_in_seconds": time.time() - start_time,
            "model_name": MODEL_NAME,
            "batched": True
        }
        response_dict[req.request_id] = result
        response_events[req.request_id].set()

def process_single_request(req: BatchRequest, model, tokenizer):
    """Process a single request."""
    start_time = time.time()

    final_prompt = apply_llama3_template(req.prompt)

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True if req.max_input else False,
        max_length=req.max_input
    ).to(model.device)

    input_seq_len = inputs.input_ids.shape[1]
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    gen_kwargs = {
        "max_new_tokens": req.max_length,
        "min_new_tokens": req.min_length,
        "do_sample": req.do_sample,
        "temperature": req.temperature if req.do_sample else 1.0,
        "top_k": req.top_k if req.top_k > 0 else None,
        "top_p": req.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": terminators,
    }

    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty
    if req.length_penalty is not None:
        gen_kwargs["length_penalty"] = req.length_penalty

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    if not req.keep_prompt:
        new_tokens = outputs[:, input_seq_len:]
        generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    else:
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    text = generated_texts[0].strip()
    result = {
        "generated_texts": [text],
        "generated_num_tokens": [len(tokenizer.encode(text))],
        "run_time_in_seconds": time.time() - start_time,
        "model_name": MODEL_NAME,
        "batched": False
    }
    response_dict[req.request_id] = result
    response_events[req.request_id].set()


# API Endpoint Definition
@app.get("/generate")
@app.get("/generate/")
async def generate_get(
    prompt: str,
    max_input: Optional[int] = None,
    max_length: int = 256, # Increased default to allow for Chain of Thought
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 0.1,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    repetition_penalty: Optional[float] = 1.1, 
    length_penalty: Optional[float] = 1.0,
    keep_prompt: bool = False,
):
    request_id = get_next_request_id()
    event = threading.Event()
    response_events[request_id] = event

    req = BatchRequest(
        request_id=request_id,
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

    request_queue.put(req)
    
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: event.wait(timeout=300))
    except:
        pass

    if request_id in response_dict:
        result = response_dict.pop(request_id)
        del response_events[request_id]
        return result
    else:
        return {"error": "Request timeout"}

@app.on_event("startup")
async def startup_event():
    """Start the batch processing worker when the server starts."""
    worker_thread = threading.Thread(target=batch_processing_worker, daemon=True)
    worker_thread.start()
    print("✓ Batch processing worker thread started")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("LLM_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)