"""
vLLM-based FastAPI server for high-throughput LLM inference.
Supports batch processing and continuous batching for better GPU utilization.

Run with: python evaluate/utils/vllm_server.py
"""

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import Optional, List
import os

app = FastAPI()

# Model configuration
MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))

# Global vLLM engine
llm = None


class GenerateRequest(BaseModel):
    """Generation request model (compatible with llm_client_generator.py)."""
    prompt: str
    max_input: Optional[int] = None
    max_length: Optional[int] = 200
    min_length: Optional[int] = 1
    do_sample: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    keep_prompt: Optional[bool] = False
    model_name: Optional[str] = None


@app.on_event("startup")
def load_model():
    """Load vLLM model on startup."""
    global llm

    print(f"Loading vLLM model from {MODEL_PATH}...")
    print(f"  Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    print(f"  GPU memory utilization: {GPU_MEMORY_UTILIZATION}")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=8192,  # Adjust based on your needs
    )

    print("✓ vLLM model loaded successfully")
    print(f"✓ Ready for high-throughput batch inference")


@app.get("/")
def index():
    """Health check endpoint."""
    return {
        "message": "vLLM server running",
        "model": MODEL_PATH,
        "status": "ready" if llm else "loading"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": llm is not None}


@app.get("/generate")
def generate_get(
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
    model_name: Optional[str] = None
):
    """Generate endpoint (GET method for compatibility)."""
    return generate(
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
        keep_prompt=keep_prompt,
        model_name=model_name
    )


@app.post("/generate")
def generate_post(request: GenerateRequest):
    """Generate endpoint (POST method)."""
    return generate(
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
        keep_prompt=request.keep_prompt,
        model_name=request.model_name
    )


def generate(
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
    model_name: Optional[str] = None
):
    """
    Generate text using vLLM (compatible with llm_client_generator.py).

    vLLM benefits:
    - Automatic batch processing
    - Continuous batching for higher throughput
    - PagedAttention for efficient memory usage
    """
    if llm is None:
        return {"error": "Model not loaded"}

    # Truncate prompt if max_input is specified
    if max_input is not None:
        tokenizer = llm.get_tokenizer()
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_input:
            tokens = tokens[:max_input]
            prompt = tokenizer.decode(tokens)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        max_tokens=max_length,
        min_tokens=min_length,
        repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
        length_penalty=length_penalty if length_penalty else 1.0,
    )

    # Generate with vLLM
    outputs = llm.generate([prompt], sampling_params)

    # Extract generated texts
    generated_texts = []
    for output in outputs:
        for completion in output.outputs:
            text = completion.text
            if keep_prompt:
                text = prompt + text
            generated_texts.append(text)

    # Return in compatible format
    return {
        "generated_texts": generated_texts,
        "model_name": model_name or "Meta-Llama-3-8B-Instruct",
        "generated_num_tokens": [len(llm.get_tokenizer().encode(text)) for text in generated_texts],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LLM_PORT", "8000"))

    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║           vLLM High-Throughput LLM Server                  ║
    ╚════════════════════════════════════════════════════════════╝

    Model: {MODEL_PATH}
    Port: {port}

    Features:
    - Continuous batching for high throughput
    - PagedAttention for memory efficiency
    - Automatic request batching

    Starting server...
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
