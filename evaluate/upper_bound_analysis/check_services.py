#!/usr/bin/env python3
"""
服务检查脚本
Check if LLM and Retriever services are running

使用方法：
    python check_services.py
"""

import requests
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config_helper import get_llm_config, get_retriever_config
    CONFIG_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load config_helper: {e}")
    CONFIG_AVAILABLE = False


def check_llm_service(host="localhost", port=8000):
    """检查 LLM 服务"""
    url = f"http://{host}:{port}/generate"

    print(f"\n{'='*70}")
    print("检查 LLM 服务")
    print(f"{'='*70}")
    print(f"URL: {url}")

    try:
        # 发送测试请求
        response = requests.get(
            url,
            params={'prompt': 'test', 'max_length': 10},
            timeout=10
        )

        if response.status_code == 200:
            print("✓ LLM 服务运行正常")
            print(f"  状态码: {response.status_code}")
            try:
                data = response.json()
                print(f"  响应: {str(data)[:100]}...")
            except:
                print(f"  响应: {response.text[:100]}...")
            return True
        else:
            print(f"✗ LLM 服务返回错误")
            print(f"  状态码: {response.status_code}")
            print(f"  响应: {response.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ LLM 服务连接失败")
        print("  原因: 无法连接到服务")
        print(f"\n  请检查:")
        print(f"  1. LLM 服务是否已启动")
        print(f"  2. 服务端口是否为 {port}")
        print(f"  3. 防火墙是否阻止了连接")
        return False

    except requests.exceptions.Timeout:
        print("✗ LLM 服务响应超时")
        print("  原因: 服务响应时间过长（>10秒）")
        return False

    except Exception as e:
        print(f"✗ LLM 服务检查失败")
        print(f"  错误: {e}")
        return False


def check_retriever_service(host="localhost", port=8001):
    """检查 Retriever 服务"""
    url = f"http://{host}:{port}/retrieve/"

    print(f"\n{'='*70}")
    print("检查 Retriever 服务")
    print(f"{'='*70}")
    print(f"URL: {url}")

    try:
        # 发送测试请求（使用与stage2相同的完整格式）
        response = requests.post(
            url,
            json={
                'retrieval_method': 'retrieve_from_elasticsearch',
                'query_text': 'test query',
                'rerank_query_text': 'test query',
                'max_hits_count': 5,
                'max_buffer_count': 20,
                'corpus_name': 'wiki',  # 默认使用wiki
                'document_type': 'title_paragraph_text',
                'retrieval_backend': 'hybrid'
            },
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            print("✓ Retriever 服务运行正常")
            print(f"  状态码: {response.status_code}")
            try:
                data = response.json()
                retrieval_results = data.get('retrieval', [])
                print(f"  响应: 成功获取 {len(retrieval_results)} 个检索结果")
                if retrieval_results:
                    print(f"  示例文档: {retrieval_results[0].get('title', 'N/A')[:50]}...")
            except Exception as e:
                print(f"  响应解析失败: {e}")
                print(f"  原始响应: {response.text[:200]}...")
            return True
        elif response.status_code == 404:
            print(f"✗ Retriever 服务路径错误 (404)")
            print(f"  URL: {url}")
            print(f"\n  可能的原因:")
            print(f"  1. 服务路径不是 /retrieve/")
            print(f"  2. 服务还未完全启动")
            print(f"\n  建议:")
            print(f"  1. 检查 Retriever 服务的 API 路径（应该是 /retrieve/）")
            print(f"  2. 查看 Retriever 服务日志")
            return False
        else:
            print(f"✗ Retriever 服务返回错误")
            print(f"  状态码: {response.status_code}")
            print(f"  响应: {response.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Retriever 服务连接失败")
        print("  原因: 无法连接到服务")
        print(f"\n  请检查:")
        print(f"  1. Retriever 服务是否已启动")
        print(f"  2. 服务端口是否为 {port}")
        print(f"  3. 防火墙是否阻止了连接")
        return False

    except requests.exceptions.Timeout:
        print("✗ Retriever 服务响应超时")
        print("  原因: 服务响应时间过长（>10秒）")
        return False

    except Exception as e:
        print(f"✗ Retriever 服务检查失败")
        print(f"  错误: {e}")
        return False


def check_data_files():
    """检查数据文件"""
    print(f"\n{'='*70}")
    print("检查数据文件")
    print(f"{'='*70}")

    datasets = ['musique', '2wikimultihopqa', 'hotpotqa', 'squad', 'trivia', 'nq']
    data_dir = "processed_data"

    # 从当前文件位置找到项目根目录
    current_file = os.path.abspath(__file__)
    upper_bound_dir = os.path.dirname(current_file)
    evaluate_dir = os.path.dirname(upper_bound_dir)
    project_root = os.path.dirname(evaluate_dir)

    found_count = 0
    missing_datasets = []

    for dataset in datasets:
        test_file = os.path.join(project_root, data_dir, dataset, "test_subsampled.jsonl")
        if os.path.exists(test_file):
            print(f"✓ {dataset}: 数据文件存在")
            found_count += 1
        else:
            print(f"✗ {dataset}: 数据文件不存在")
            print(f"  路径: {test_file}")
            missing_datasets.append(dataset)

    print(f"\n总结: 找到 {found_count}/{len(datasets)} 个数据集")

    if missing_datasets:
        print(f"\n缺失的数据集: {', '.join(missing_datasets)}")
        print("\n提示: 运行数据预处理脚本生成测试数据")
        return False

    return True


def main():
    """主函数"""
    print("\n" + "="*70)
    print("上限测试服务检查工具")
    print("Upper Bound Analysis - Service Check")
    print("="*70)

    # 从配置文件读取服务地址
    llm_host = "localhost"
    llm_port = 8000
    retriever_host = "localhost"
    retriever_port = 8001

    if CONFIG_AVAILABLE:
        try:
            llm_config = get_llm_config()
            llm_host = llm_config.get('host', 'localhost')
            llm_port = llm_config.get('port', 8000)

            retriever_config = get_retriever_config()
            retriever_host = retriever_config.get('host', 'localhost')
            retriever_port = retriever_config.get('port', 8001)

            print(f"\n从配置文件读取:")
            print(f"  LLM:       {llm_host}:{llm_port}")
            print(f"  Retriever: {retriever_host}:{retriever_port}")
        except Exception as e:
            print(f"\nWarning: 无法读取配置文件，使用默认值")
            print(f"  错误: {e}")

    # 检查服务
    llm_ok = check_llm_service(llm_host, llm_port)
    retriever_ok = check_retriever_service(retriever_host, retriever_port)
    data_ok = check_data_files()

    # 总结
    print(f"\n{'='*70}")
    print("检查总结")
    print(f"{'='*70}")

    services_status = []
    if llm_ok:
        services_status.append("✓ LLM 服务正常")
    else:
        services_status.append("✗ LLM 服务异常")

    if retriever_ok:
        services_status.append("✓ Retriever 服务正常")
    else:
        services_status.append("✗ Retriever 服务异常")

    if data_ok:
        services_status.append("✓ 数据文件完整")
    else:
        services_status.append("✗ 数据文件缺失")

    for status in services_status:
        print(status)

    # 给出建议
    if llm_ok and retriever_ok and data_ok:
        print("\n" + "="*70)
        print("✓ 所有检查通过！可以开始运行测试")
        print("="*70)
        print("\n运行测试:")
        print("  python run_all_tests.py")
        print("\n或查看快速参考:")
        print("  cat QUICK_REFERENCE.md")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠ 部分检查未通过")
        print("="*70)

        if not llm_ok:
            print("\nLLM 服务问题:")
            print("  1. 确保 LLM 服务已启动")
            print("  2. 检查服务端口配置")
            print("  3. 查看 LLM 服务日志")

        if not retriever_ok:
            print("\nRetriever 服务问题:")
            print("  1. 确保 Retriever 服务已启动")
            print("  2. 检查服务端口配置")
            print("  3. 如果是 404 错误，检查 API 路径是否为 /search")
            print("  4. 可以跳过 Retriever 测试: python run_all_tests.py --skip-retriever")

        if not data_ok:
            print("\n数据文件问题:")
            print("  1. 运行数据预处理脚本")
            print("  2. 或者只测试已有的数据集")

        return 1


if __name__ == '__main__':
    sys.exit(main())
