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
    """Generation request model."""
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 1.0
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
    max_length: int = 200,
    temperature: float = 1.0,
    model_name: str = None
):
    """Generate endpoint (GET method for compatibility)."""
    return generate(prompt, max_length, temperature, model_name)


@app.post("/generate")
def generate_post(request: GenerateRequest):
    """Generate endpoint (POST method)."""
    return generate(
        request.prompt,
        request.max_length,
        request.temperature,
        request.model_name
    )


def generate(
    prompt: str,
    max_length: int = 200,
    temperature: float = 1.0,
    model_name: str = None
):
    """
    Generate text from prompt.

    Args:
        prompt: Input prompt
        max_length: Maximum new tokens to generate
        temperature: Sampling temperature
        model_name: Model name (for compatibility)

    Returns:
        dict with generated_texts and model_name
    """
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    generated_only = generated_text[len(prompt):]

    return {
        "generated_texts": [generated_only],
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
