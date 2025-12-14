"""
Custom FastAPI server for Llama 3-8B (fallback if vLLM not available).
Run with: uvicorn evaluate.utils.llama_fastapi_server:app --host localhost --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None

# Model configuration (should match config.yaml)
MODEL_PATH = "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"


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
    """Load model on startup."""
    global model, tokenizer

    print(f"Loading Llama model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Llama model loaded successfully")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


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
    """Generate endpoint (GET method for compatibility with llm_client_generator.py)."""
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
    Generate text from prompt (compatible with llm_client_generator.py).

    Args:
        prompt: Input prompt
        max_input: Maximum input length (if provided, truncate prompt)
        max_length: Maximum new tokens to generate
        min_length: Minimum new tokens to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        num_return_sequences: Number of sequences to generate
        repetition_penalty: Repetition penalty
        length_penalty: Length penalty
        keep_prompt: If True, include prompt in output
        model_name: Model name (for compatibility)

    Returns:
        dict with generated_texts and model_name
    """
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    # Truncate prompt if max_input is specified
    if max_input is not None:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = inputs.to(model.device)

    # Prepare generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_length,
        "min_new_tokens": min_length,
        "do_sample": do_sample or temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Add optional penalties if specified
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if length_penalty is not None:
        gen_kwargs["length_penalty"] = length_penalty

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode all sequences
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)

        # Remove prompt from output unless keep_prompt is True
        if not keep_prompt:
            generated_text = generated_text[len(prompt):].strip()

        generated_texts.append(generated_text)

    # Extract model name from path if not provided
    # Return format compatible with llm_client_generator validation
    response_model_name = model_name if model_name else "Meta-Llama-3-8B-Instruct"
    if "/" in response_model_name:
        response_model_name = response_model_name.split("/")[1]

    return {
        "generated_texts": generated_texts,
        "model_name": response_model_name
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
