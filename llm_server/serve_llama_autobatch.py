"""
FastAPI LLM Server with Automatic Request Batching for Llama models.

Features:
- Automatic request batching for better GPU utilization
- Transparent to client - no code changes needed
- Dynamic batch sizing based on queue depth
- Compatible with existing llm_client_generator.py

How it works:
1. Incoming requests are queued
2. Background worker collects requests every 50ms or when batch size reached
3. Processes batch with single GPU forward pass
4. Returns results to individual requests

Usage:
    export MODEL_PATH="/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
    export LLM_PORT=8000
    export BATCH_SIZE=8  # Max batch size (default: 8)
    export BATCH_TIMEOUT_MS=50  # Wait time in ms (default: 50)
    pixi run python llm_server/serve_llama_autobatch.py
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
    将原始 prompt 包装成 Llama-3 的 Chat Template 格式。
    根据不同的策略类型使用不同的 System Prompt。

    策略识别：
    - 如果包含 "reasoning step-by-step"：IRCoT CoT 推理阶段，允许详细推理
    - 如果包含 "Answer the following question." 但不包含 "step-by-step"：Direct QA 模式
    - 其他：默认精确简短模式
    """

    # 检测是否为 IRCoT CoT 推理阶段（M 策略）
    is_ircot_cot = "reasoning step-by-step" in raw_content or "reason step-by-step" in raw_content

    # 检测是否为 Direct QA 阶段
    is_direct_qa = "Answer the following question" in raw_content and not is_ircot_cot

    if is_ircot_cot:
        # IRCoT CoT 模式：逐句生成事实陈述，系统会用最后的事实句作为下一次检索的查询
        # 关键机制：
        # 1. 生成的句子会被用 spacy 分句，只取第一个句子
        # 2. 非推理句（事实陈述）会被提取作为下一次检索的查询
        # 3. 推理句（以 "So", "Thus", "Therefore" 开头）不会用于检索，表示找到答案
        system_prompt = (
            "You are a multi-step reasoning agent that answers complex questions by gathering information step-by-step.\n\n"
            "IMPORTANT CONTEXT:\n"
            "- The 'A:' field below shows the reasoning chain SO FAR (may be empty at first)\n"
            "- The examples show COMPLETE chains (built through multiple iterations)\n"
            "- You will be called multiple times - each time you add ONE new sentence\n\n"
            "YOUR TASK:\n"
            "Based on the question, the provided documents, and the reasoning so far, generate EXACTLY ONE new sentence.\n\n"
            "DECISION LOGIC:\n"
            "1. IF the documents contain the FINAL ANSWER → Write: 'So the answer is: [answer]'\n"
            "2. IF the documents contain a KEY FACT or BRIDGE ENTITY needed for next step → State that fact clearly\n"
            "   - Your sentence will be used as a SEARCH QUERY to retrieve more documents\n"
            "   - Include specific entities, names, dates, or keywords (not generic words)\n"
            "   - Example: 'Neville A. Stanton works at University of Southampton.' (good - has entity)\n"
            "   - NOT: 'We need to find more information.' (bad - no searchable keywords)\n"
            "3. IF the documents are irrelevant → State what specific information is MISSING\n\n"
            "CRITICAL RULES:\n"
            "- Output ONLY ONE sentence (not a paragraph)\n"
            "- Do NOT repeat what is already in 'A:'\n"
            "- Do NOT start with 'So'/'Thus'/'Therefore' unless you have the final answer\n"
            "- Make your sentence rich in specific entities and keywords (it will be used as a search query)\n"
            "- Base your statement ONLY on the provided documents\n\n"
            "EXAMPLE:\n"
            "Q: When was Neville A. Stanton's employer founded?\n"
            "Call 1 - A: [empty], Docs: [info about Stanton] → Output: 'Neville A. Stanton works at University of Southampton.'\n"
            "Call 2 - A: Neville A. Stanton works..., Docs: [info about Southampton] → Output: 'University of Southampton was founded in 1862.'\n"
            "Call 3 - A: Neville A. Stanton works... University of Southampton was founded..., Docs: [...] → Output: 'So the answer is: 1862.'"
        )
    elif is_direct_qa:
        # Direct QA 模式：简短直接的答案（IRCoT 最终回答阶段或其他 Direct QA 策略）
        system_prompt = (
            "You are a precise QA assistant.\n\n"
            "Rules:\n"
            "1. Read the provided context and answer the question directly.\n"
            "2. Output ONLY the answer - be concise and specific.\n"
            "3. Do NOT include phrases like 'The answer is...' or 'Based on...'.\n"
            "4. Base your answer ONLY on the provided context.\n"
            "5. If the answer is not in the context, output 'I don't know'."
        )
    else:
        # 默认模式：用于其他策略，保持原有的精确简短逻辑
        system_prompt = (
            "You are a precise QA assistant. "
            "Read the provided context and answer the user's question.\n\n"
            "Rules:\n"
            "1. Answer based ONLY on the context. Do not use your internal knowledge.\n"
            "2. Be extremely concise. Output ONLY the answer entity or short phrase.\n"
            "3. Do NOT write full sentences. Do NOT say 'The answer is...'.\n"
            "4. If the answer is not in the context, output exactly 'I don't know'."
        )

    # Llama-3 特殊 Token 定义 (为了代码可读性提取出来)
    header_start = "<|start_header_id|>"
    header_end = "<|end_header_id|>"
    eot = "<|eot_id|>"

    # 构建 Llama-3 格式的 prompt
    formatted_prompt = (
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL: Use left padding for batched generation
    # Right padding causes misalignment between prompt and generated tokens
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
    Collects requests from queue and processes them in batches.
    """
    model, tokenizer = get_model_and_tokenizer()

    print("✓ Batch processing worker started")

    while True:
        # Collect batch
        batch = []
        start_collect = time.time()
        timeout_seconds = BATCH_TIMEOUT_MS / 1000.0

        # Get first request (blocking)
        try:
            first_req = request_queue.get(timeout=1.0)
            batch.append(first_req)
        except:
            continue  # No requests, try again

        # Collect more requests up to batch size or timeout
        while len(batch) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_collect
            remaining_timeout = max(0.001, timeout_seconds - elapsed)

            try:
                req = request_queue.get(timeout=remaining_timeout)
                batch.append(req)
            except:
                break  # Timeout or no more requests

        # Process batch
        if batch:
            process_batch(batch, model, tokenizer)


def process_batch(batch: List[BatchRequest], model, tokenizer):
    """Process a batch of requests."""
    batch_size = len(batch)

    if batch_size == 1:
        # Single request - optimize for latency
        process_single_request(batch[0], model, tokenizer)
    else:
        # Batch processing - optimize for throughput
        try:
            # Group by generation parameters (can only batch if params match)
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

            # Process each parameter group
            for param_key, group in param_groups.items():
                if len(group) == 1:
                    process_single_request(group[0], model, tokenizer)
                else:
                    process_batch_group(group, model, tokenizer)

        except Exception as e:
            # On error, fall back to processing individually
            print(f"Batch processing error: {e}, falling back to individual processing")
            for req in batch:
                process_single_request(req, model, tokenizer)

def process_batch_group(group: List[BatchRequest], model, tokenizer):
    """Process a group of requests with same parameters as a true batch."""
    start_time = time.time()

    # Extract prompts and apply Llama-3 template
    prompts = [apply_llama3_template(req.prompt) for req in group]
    sample_req = group[0]

    # Batch tokenization
    # padding=True 会自动填充到该 Batch 中最长的序列长度
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True if sample_req.max_input else False,
        max_length=sample_req.max_input,
        padding=True,
        return_attention_mask=True
    ).to(model.device)

    # 获取输入 Tensor 的物理宽度 (Input Length)
    # 这是最关键的一步：无论有多少 Padding，生成的 Token 都在这个长度之后
    input_seq_len = inputs.input_ids.shape[1]

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": sample_req.max_length,
        "min_new_tokens": sample_req.min_length,
        "do_sample": sample_req.do_sample,
        "temperature": sample_req.temperature if sample_req.do_sample else 1.0,
        "top_k": sample_req.top_k if sample_req.top_k > 0 else None,
        "top_p": sample_req.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": inputs.attention_mask,
    }

    if sample_req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = sample_req.repetition_penalty
    if sample_req.length_penalty is not None:
        gen_kwargs["length_penalty"] = sample_req.length_penalty

    # Batch generate
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **gen_kwargs)

    # Process each result
    # 只需要解码新生成的 tokens
    # outputs 的形状是 [batch_size, input_seq_len + new_tokens]
    # 我们只取 [:, input_seq_len:]
    new_tokens = outputs[:, input_seq_len:]
    
    # 批量解码提高效率
    generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Process each result with token counts
    for i, req in enumerate(group):
        text = generated_texts[i].strip()
        
        # 如果需要保留 Prompt (极为罕见，通常在 Chat 模式不需要)，则拼接回去
        if req.keep_prompt:
            text = req.prompt + text

        num_tokens = len(tokenizer.encode(text))

        result = {
            "generated_texts": [text],
            "generated_num_tokens": [num_tokens],
            "run_time_in_seconds": time.time() - start_time,
            "model_name": MODEL_NAME,
            "batched": True,
            "batch_size": len(group)
        }

        # Store result and signal completion
        response_dict[req.request_id] = result
        response_events[req.request_id].set()

def process_single_request(req: BatchRequest, model, tokenizer):
    """Process a single request (fallback or single-item batch)."""
    start_time = time.time()

    # Apply Llama-3 template and tokenize
    final_prompt = apply_llama3_template(req.prompt)

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True if req.max_input else False,
        max_length=req.max_input,
        padding=False # 单条不需要 Padding
    ).to(model.device)

    input_seq_len = inputs.input_ids.shape[1]

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": req.max_length,
        "min_new_tokens": req.min_length,
        "do_sample": req.do_sample,
        "temperature": req.temperature if req.do_sample else 1.0,
        "top_k": req.top_k if req.top_k > 0 else None,
        "top_p": req.top_p,
        "num_return_sequences": req.num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty
    if req.length_penalty is not None:
        gen_kwargs["length_penalty"] = req.length_penalty

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode
    if not req.keep_prompt:
        # 只取新生成的 Tokens 进行解码
        new_tokens = outputs[:, input_seq_len:]
        generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    else:
        # 解码全部
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Clean up
    generated_texts = [text.strip() for text in generated_texts]
    generated_num_tokens = [len(tokenizer.encode(text)) for text in generated_texts]

    result = {
        "generated_texts": generated_texts,
        "generated_num_tokens": generated_num_tokens,
        "run_time_in_seconds": time.time() - start_time,
        "model_name": MODEL_NAME,
        "batched": False
    }

    # Store result and signal completion
    response_dict[req.request_id] = result
    response_events[req.request_id].set()


class GenerateRequest(BaseModel):
    """Single generation request."""
    prompt: str
    max_input: Optional[int] = None
    max_length: int = 64  # Reduced for concise answers (was 256)
    min_length: int = 1
    do_sample: bool = False  # Keep greedy decoding for consistency
    temperature: float = 0.1  # Further lowered for more deterministic, focused outputs
    top_k: int = 50
    top_p: float = 0.95  # Slightly reduced from 1.0 for better quality
    num_return_sequences: int = 1
    repetition_penalty: Optional[float] = 1.2  # Added to prevent repetition
    length_penalty: Optional[float] = 0.8  # Slight penalty for longer outputs
    keep_prompt: bool = False


@app.on_event("startup")
async def startup_event():
    """Start background batch processing worker."""
    # Preload model
    get_model_and_tokenizer()

    # Start batch worker thread
    worker_thread = threading.Thread(target=batch_processing_worker, daemon=True)
    worker_thread.start()

    print("✓ Automatic batching enabled!")


@app.get("/")
async def index():
    """Health check endpoint."""
    return {
        "message": "Llama Auto-Batch Server running",
        "model": MODEL_NAME,
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": BATCH_TIMEOUT_MS
    }


@app.get("/generate")
@app.get("/generate/")
async def generate_get(
    prompt: str,
    max_input: Optional[int] = None,
    max_length: int = 64,  # Reduced for concise answers
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 0.1,  # Very low for focused, deterministic outputs
    top_k: int = 50,
    top_p: float = 0.95,  # Slightly reduced for better quality
    num_return_sequences: int = 1,
    repetition_penalty: Optional[float] = 1.2,  # Added to prevent repetition
    length_penalty: Optional[float] = 0.8,  # Slight penalty for longer outputs
    keep_prompt: bool = False,
):
    """Generate endpoint (GET method - compatible with llm_client_generator.py)."""
    # Create request
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

    # Queue request
    request_queue.put(req)
    
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(None, lambda: event.wait(timeout=300))
    except:
        pass

    # Get and return response
    if request_id in response_dict:
        result = response_dict.pop(request_id)
        del response_events[request_id]
        return result
    else:
        return {"error": "Request timeout"}


if __name__ == "__main__":
    import uvicorn

    print(f"""
╔════════════════════════════════════════════════════════════╗
║      Llama Auto-Batch Inference Server                     ║
╚════════════════════════════════════════════════════════════╝

Model: {MODEL_PATH}
Port: {os.environ.get("LLM_PORT", "8000")}

Auto-Batching Configuration:
  Max batch size: {MAX_BATCH_SIZE}
  Batch timeout: {BATCH_TIMEOUT_MS}ms

How it works:
  - Requests are automatically queued
  - Processed in batches every {BATCH_TIMEOUT_MS}ms or when {MAX_BATCH_SIZE} requests collected
  - Transparent to client - no code changes needed!

Loading model...
""")

    port = int(os.environ.get("LLM_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
