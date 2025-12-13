"""
Service health checker for pre-flight validation.
"""

import requests
import time


class ServiceChecker:
    """Validate that required services are running."""

    def __init__(self, config):
        self.config = config
        self.retriever_host = config['retriever']['host']
        self.retriever_port = config['retriever']['port']
        self.llm_host = config['llm']['server_host']
        self.llm_port = config['llm']['server_port']

    def check_all(self):
        """Run all service checks."""
        print("\n" + "="*60)
        print("Running Service Health Checks")
        print("="*60)

        checks = [
            ('Retriever Service', self.check_retriever),
            ('LLM Server', self.check_llm_server),
        ]

        all_passed = True
        for name, check_func in checks:
            print(f"Checking {name}...", end=' ')
            try:
                check_func()
                print("✓ OK")
            except Exception as e:
                print(f"✗ FAILED: {e}")
                all_passed = False

        print("="*60)

        if not all_passed:
            raise RuntimeError(
                "Service checks failed. Please start required services.\n"
                f"Retriever: http://{self.retriever_host}:{self.retriever_port}\n"
                f"LLM Server: http://{self.llm_host}:{self.llm_port}"
            )

        print("✓ All services are ready\n")

    def check_retriever(self):
        """Check if retriever service is running."""
        url = f"http://{self.retriever_host}:{self.retriever_port}/health"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Retriever returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot connect to retriever: {e}")

    def check_llm_server(self):
        """Check if LLM server is running."""
        # Try health endpoint first
        health_url = f"http://{self.llm_host}:{self.llm_port}/health"

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass

        # Fallback: try a simple generation request
        gen_url = f"http://{self.llm_host}:{self.llm_port}/generate"

        try:
            params = {
                'prompt': 'test',
                'max_length': 1,
                'model_name': self.config['llm']['model_name']
            }
            response = requests.get(gen_url, params=params, timeout=10)

            if response.status_code != 200:
                raise RuntimeError(f"LLM server returned status {response.status_code}")

            # Verify model name in response
            result = response.json()
            if 'model_name' in result:
                expected = self.config['llm']['model_name'].split('/')[-1]
                actual = result['model_name'].split('/')[-1]

                if expected.lower() not in actual.lower():
                    raise RuntimeError(
                        f"Wrong model loaded. Expected {expected}, got {actual}"
                    )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot connect to LLM server: {e}")

    def wait_for_service(self, url: str, timeout: int = 300, service_name: str = "Service"):
        """
        Wait for a service to become available.

        Args:
            url: Service URL to check
            timeout: Maximum wait time in seconds
            service_name: Name of service for logging
        """
        start_time = time.time()

        print(f"Waiting for {service_name} at {url}...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ {service_name} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(5)
            elapsed = int(time.time() - start_time)
            print(f"  Still waiting... ({elapsed}s elapsed)")

        raise TimeoutError(f"{service_name} failed to start within {timeout}s")
