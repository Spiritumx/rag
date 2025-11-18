import logging
import time
import os
from functools import lru_cache

from openai import OpenAI
from diskcache import Cache
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit


logger = logging.getLogger(__name__)

# Local response cache
cache = Cache(os.path.expanduser("~/.cache/gpt_calls"))


def get_openai_client(api_key=None, base_url=None):
    """Create a configurable OpenAI client"""
    # Priority: passed api_key > environment variable > default
    final_api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-BZQdNZeSwyih3TKpD95fDd83A90e4556A95f7eB7D489C36b"
    final_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.gpt.ge/v1/"
    
    return OpenAI(
        api_key=final_api_key,
        base_url=final_base_url
    )


@cache.memoize()
def cached_openai_call(api_key, base_url, model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, n):
    """Cached ChatCompletion request - uses api_key and base_url instead of client to avoid serialization issues"""
    client = get_openai_client(api_key, base_url)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
    )


def openai_call(client:OpenAI, api_key, base_url, model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, n):
    """Live call or cached call depending on temperature"""
    if temperature == 0:
        return cached_openai_call(api_key, base_url, model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, n)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
    )


@lru_cache(maxsize=1)
def get_gpt_tokenizer():
    """GPT tokenizer"""
    from transformers import GPT2Tokenizer
    return GPT2Tokenizer.from_pretrained("gpt2")


class GPTGenerator:

    def __init__(
        self,
        model="gpt-4o-mini",
        engine=None,  # Backward compatibility with old configs
        api_key=None,
        base_url=None,
        temperature=0,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
        retry_after_n_seconds=3,
        n=1,
        remove_method="first",
    ):
        self.client = get_openai_client(api_key, base_url)
        # Support both 'engine' (old) and 'model' (new) parameter names
        self.model = engine if engine is not None else model
        # Save api_key and base_url for caching (client object is not serializable)
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.retry_after_n_seconds = retry_after_n_seconds
        self.n = n
        self.remove_method = remove_method

        # context limits (adjustable)
        # Set context limits based on model capabilities
        if "gpt-4o" in model.lower():
            # GPT-4o models support up to 128k tokens
            self.model_tokens_limit = 120000
        elif "gpt-4" in model.lower():
            # GPT-4 models support up to 8k tokens (some variants up to 32k)
            self.model_tokens_limit = 8000
        elif "gpt-3.5" in model.lower() or "gpt-35" in model.lower():
            # GPT-3.5 models support up to 16k tokens
            self.model_tokens_limit = 16000
        elif "gpt" in model.lower():
            # Other GPT models, use a conservative limit
            self.model_tokens_limit = 8000
        else:
            # Default for other models
            self.model_tokens_limit = 4000


    def generate_text_sequence(self, prompt):
        """Main method to generate response sequence"""

        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name="gpt2",
            last_is_test_example=True,
        )

        messages = [{"role": "user", "content": prompt}]
        success = False
        response = None
        has_retried_for_none = False

        for attempt in range(10):  # Retry loop safety
            try:
                response = openai_call(
                    client=self.client,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop,
                    n=self.n,
                )
                
                # Check if response has None values
                has_none = False
                if response and response.choices:
                    for c in response.choices:
                        if c.message is None or c.message.content is None:
                            has_none = True
                            break
                
                # If we detect None and haven't retried yet, retry once
                if has_none and not has_retried_for_none:
                    logger.warning(f"Detected None in API response (attempt {attempt + 1}), retrying once...")
                    has_retried_for_none = True
                    time.sleep(self.retry_after_n_seconds)
                    continue  # Retry the request
                
                success = True
                break

            except Exception as e:
                logger.warning(f"OpenAI request failed: {e}, retrying in {self.retry_after_n_seconds}s...")
                time.sleep(self.retry_after_n_seconds)

        if not success or response is None:
            raise RuntimeError("Failed to retrieve model output after retries.")

        # Extract results
        outputs = []
        for idx, c in enumerate(response.choices):
            if c.message is None:
                logger.warning(f"Choice {idx} has no message, skipping")
                continue
            content = c.message.content if c.message.content is not None else ""
            outputs.append((content.strip(), idx))
        
        if not outputs:
            logger.warning("No valid outputs from API response, returning empty result")
            return [("", 0)]
        
        return sorted(outputs, key=lambda x: x[1])
