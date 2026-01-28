"""
快速验证消融实验检索配置是否正确
只测试每个模型的检索连通性，不运行完整实验
"""

import os
import sys
import requests

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# 模型配置
MODELS = {
    'A': {'port': 8002, 'desc': '完整系统 (Adaptive)'},
    'B': {'port': 8002, 'desc': 'w/o ToT'},
    'C': {'port': 8002, 'desc': 'w/o Cascade'},
    'D': {'port': 8001, 'desc': 'w/o Adaptive (基线检索器)'},
}

def test_retriever(host: str, port: int, model_id: str) -> bool:
    """测试检索服务连通性"""
    url = f"http://{host}:{port}/retrieve/"

    try:
        response = requests.post(
            url,
            json={
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": "Who is the president of United States?",
                "max_hits_count": 3,
                "max_buffer_count": 10,
                "corpus_name": "wiki",
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            hits = data.get('retrieval', [])
            if hits:
                print(f"  ✓ 检索成功，返回 {len(hits)} 条结果")
                print(f"    示例: {hits[0].get('title', 'N/A')[:50]}...")
                return True
            else:
                print(f"  ⚠ 连接成功但无结果返回")
                return False
        else:
            print(f"  ✗ HTTP 错误: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"  ✗ 连接失败 - 服务未运行或端口错误")
        return False
    except requests.exceptions.Timeout:
        print(f"  ✗ 请求超时")
        return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False


def test_subprocess_env():
    """测试子进程环境变量传递"""
    import subprocess
    import tempfile

    # 创建测试脚本
    test_script = """
import os
print(f"RETRIEVER_HOST={os.environ.get('RETRIEVER_HOST', 'NOT_SET')}")
print(f"RETRIEVER_PORT={os.environ.get('RETRIEVER_PORT', 'NOT_SET')}")
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # 测试环境变量传递
        env = os.environ.copy()
        env['RETRIEVER_HOST'] = 'test_host'
        env['RETRIEVER_PORT'] = '9999'

        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            env=env
        )

        output = result.stdout
        if 'RETRIEVER_HOST=test_host' in output and 'RETRIEVER_PORT=9999' in output:
            print("  ✓ 环境变量传递正常")
            return True
        else:
            print(f"  ✗ 环境变量传递失败: {output}")
            return False
    finally:
        os.remove(script_path)


def main():
    host = "localhost"

    print("=" * 60)
    print("消融实验快速验证")
    print("=" * 60)

    # 1. 测试环境变量传递
    print("\n[1] 测试子进程环境变量传递...")
    test_subprocess_env()

    # 2. 测试各模型的检索服务
    print("\n[2] 测试检索服务连通性...")
    print("-" * 60)

    results = {}
    ports_tested = set()

    for model_id, config in MODELS.items():
        port = config['port']
        desc = config['desc']

        print(f"\nModel {model_id}: {desc}")
        print(f"  端口: {port}")

        # 避免重复测试相同端口
        if port in ports_tested:
            print(f"  (端口 {port} 已测试，跳过)")
            results[model_id] = results[[k for k, v in MODELS.items() if v['port'] == port and k in results][0]]
            continue

        results[model_id] = test_retriever(host, port, model_id)
        ports_tested.add(port)

    # 3. 总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)

    all_pass = True
    for model_id in ['A', 'B', 'C', 'D']:
        config = MODELS[model_id]
        status = "✓ PASS" if results.get(model_id) else "✗ FAIL"
        if not results.get(model_id):
            all_pass = False
        print(f"  Model {model_id} (port {config['port']}): {status}")

    print("-" * 60)

    if all_pass:
        print("✓ 所有检索服务正常，可以运行消融实验")
    else:
        print("✗ 部分检索服务不可用，请检查:")
        if not results.get('D'):
            print("  - Model D 需要端口 8001 的基线检索器")
            print("    启动命令: python retriever_server.py --port 8001")
        if not results.get('A'):
            print("  - Model A/B/C 需要端口 8002 的 V2 检索器")
            print("    启动命令: python retriever_server_v2.py --port 8002")

    print("=" * 60)

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
