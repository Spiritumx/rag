"""
LLM Backend Wrapper

支持多种 LLM Backend：
- Local Llama (本地 Llama-3-8B)
- OpenAI GPT-4o
- DeepSeek-V3
- 其他兼容 OpenAI API 的服务

用于对比不同 LLM 的逻辑规划能力
"""

import requests
import os
from typing import Dict, Any, Optional
import json


class LLMBackend:
    """LLM Backend 基类"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """生成文本"""
        raise NotImplementedError


class LocalLlamaBackend(LLMBackend):
    """本地 Llama Backend"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        super().__init__("Llama-3-8B-Local")
        self.url = f"http://{host}:{port}/generate"

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """调用本地 Llama 服务"""
        try:
            response = requests.get(
                self.url,
                params={
                    'prompt': prompt,
                    'max_length': max_tokens,
                    'temperature': temperature
                },
                timeout=40
            )

            if response.status_code != 200:
                return f"Error: LLM service failed (status {response.status_code})"

            llm_output = response.json()

            # 提取文本
            text = ""
            if 'generated_texts' in llm_output:
                texts = llm_output['generated_texts']
                text = texts[0] if isinstance(texts, list) and len(texts) > 0 else ""
            elif 'text' in llm_output:
                text = llm_output['text']
            elif 'generated_text' in llm_output:
                text = llm_output['generated_text']

            return text.strip()

        except Exception as e:
            return f"Error: {str(e)}"


class OpenAIBackend(LLMBackend):
    """OpenAI GPT Backend"""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = "https://api.openai.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """调用 OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                self.api_base,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                return f"Error: OpenAI API failed (status {response.status_code}): {response.text}"

            result = response.json()
            text = result['choices'][0]['message']['content']

            return text.strip()

        except Exception as e:
            return f"Error: {str(e)}"


class DeepSeekBackend(LLMBackend):
    """DeepSeek Backend"""

    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_base = "https://api.deepseek.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """调用 DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                self.api_base,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                return f"Error: DeepSeek API failed (status {response.status_code}): {response.text}"

            result = response.json()
            text = result['choices'][0]['message']['content']

            return text.strip()

        except Exception as e:
            return f"Error: {str(e)}"


class CustomOpenAICompatibleBackend(LLMBackend):
    """自定义 OpenAI 兼容 API Backend"""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: Optional[str] = None
    ):
        super().__init__(model)
        self.api_key = api_key or "dummy"
        self.api_base = api_base

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """调用自定义 OpenAI 兼容 API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                self.api_base,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                return f"Error: API failed (status {response.status_code}): {response.text}"

            result = response.json()
            text = result['choices'][0]['message']['content']

            return text.strip()

        except Exception as e:
            return f"Error: {str(e)}"


def create_backend(backend_type: str, **kwargs) -> LLMBackend:
    """
    创建 LLM Backend

    Args:
        backend_type: Backend 类型 (local_llama, gpt4, deepseek, custom)
        **kwargs: Backend 特定参数

    Returns:
        LLMBackend 实例
    """
    if backend_type == "local_llama":
        return LocalLlamaBackend(
            host=kwargs.get('host', 'localhost'),
            port=kwargs.get('port', 8000)
        )

    elif backend_type == "gpt4":
        return OpenAIBackend(
            model=kwargs.get('model', 'gpt-4o'),
            api_key=kwargs.get('api_key')
        )

    elif backend_type == "deepseek":
        return DeepSeekBackend(
            model=kwargs.get('model', 'deepseek-chat'),
            api_key=kwargs.get('api_key')
        )

    elif backend_type == "custom":
        return CustomOpenAICompatibleBackend(
            model=kwargs.get('model', 'unknown'),
            api_base=kwargs.get('api_base'),
            api_key=kwargs.get('api_key')
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


if __name__ == '__main__':
    # 测试不同 Backend
    print("Testing LLM Backends...\n")

    # 测试本地 Llama
    print("1. Testing Local Llama...")
    try:
        llama = create_backend("local_llama", host="localhost", port=8000)
        response = llama.generate("What is 2+2?", max_tokens=50)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # 测试 GPT-4（如果有 API key）
    if os.environ.get("OPENAI_API_KEY"):
        print("2. Testing GPT-4...")
        try:
            gpt4 = create_backend("gpt4")
            response = gpt4.generate("What is 2+2?", max_tokens=50)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    # 测试 DeepSeek（如果有 API key）
    if os.environ.get("DEEPSEEK_API_KEY"):
        print("3. Testing DeepSeek...")
        try:
            deepseek = create_backend("deepseek")
            response = deepseek.generate("What is 2+2?", max_tokens=50)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
