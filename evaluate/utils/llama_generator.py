"""
Llama 3-8B LLM server manager.
"""

import subprocess
import time
import requests
import os


class LlamaServerManager:
    """Manage Llama 3-8B LLM server using vLLM or transformers."""

    def __init__(self, config):
        self.model_path = config['llm']['model_path']
        self.host = config['llm']['server_host']
        self.port = config['llm']['server_port']
        self.process = None

    def start_server(self, use_vllm=True):
        """
        Start Llama server using vLLM (preferred) or transformers.

        Args:
            use_vllm: If True, use vLLM; otherwise use custom FastAPI server

        Example vLLM command:
        python -m vllm.entrypoints.api_server \
            --model /root/autodl-tmp/model/Meta-Llama-3-8B-Instruct \
            --host localhost \
            --port 8000 \
            --dtype auto
        """
        print(f"\nStarting Llama server at {self.host}:{self.port}...")

        if use_vllm:
            cmd = [
                "python", "-m", "vllm.entrypoints.api_server",
                "--model", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "--dtype", "auto",
                "--max-model-len", "4096"
            ]
        else:
            # Use custom FastAPI server (evaluate/utils/llama_fastapi_server.py)
            server_script = os.path.join(
                os.path.dirname(__file__),
                'llama_fastapi_server.py'
            )
            cmd = [
                "uvicorn",
                "evaluate.utils.llama_fastapi_server:app",
                "--host", self.host,
                "--port", str(self.port)
            ]

        print(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        self.wait_for_server()

    def wait_for_server(self, timeout=300):
        """
        Wait for LLM server to be ready.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()

        # Try health endpoint first
        health_url = f"http://{self.host}:{self.port}/health"
        gen_url = f"http://{self.host}:{self.port}/generate"

        print(f"Waiting for server to be ready...")

        while time.time() - start_time < timeout:
            try:
                # Try health endpoint
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ Llama server ready at {self.host}:{self.port}")
                    return True
            except requests.exceptions.RequestException:
                pass

            try:
                # Try generation endpoint as fallback
                response = requests.get(
                    gen_url,
                    params={'prompt': 'test', 'max_length': 1},
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"✓ Llama server ready at {self.host}:{self.port}")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(5)
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:
                print(f"  Still waiting... ({elapsed}s elapsed)")

        raise TimeoutError(f"Llama server failed to start within {timeout}s")

    def stop_server(self):
        """Stop the LLM server."""
        if self.process:
            print(f"Stopping Llama server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✓ Llama server stopped")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
