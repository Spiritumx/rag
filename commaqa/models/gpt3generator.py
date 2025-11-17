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
        self.model_tokens_limit = 120000 if "gpt-4o" in model else 4000


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
                success = True
                break

            except Exception as e:
                logger.warning(f"OpenAI request failed: {e}, retrying in {self.retry_after_n_seconds}s...")
                time.sleep(self.retry_after_n_seconds)

        if not success:
            raise RuntimeError("Failed to retrieve model output after retries.")

        # Extract results
        outputs = [(c.message.content.strip(), idx) for idx, c in enumerate(response.choices)]
        return sorted(outputs, key=lambda x: x[1])
