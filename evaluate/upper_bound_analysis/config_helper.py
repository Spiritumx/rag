"""
配置加载辅助工具

为上限测试模块提供统一的配置加载，支持：
1. 从 upper_bound_analysis/config.yaml 加载（推荐，独立配置）
2. 从 evaluate/config.yaml 加载（备用，主配置）
3. 命令行参数覆盖
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_upper_bound_config(config_path: str = "evaluate/upper_bound_analysis/config.yaml") -> Dict[str, Any]:
    """
    加载上限测试配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        # 如果统一配置不存在，尝试从主配置加载
        fallback_path = "evaluate/config.yaml"
        if os.path.exists(fallback_path):
            print(f"Warning: {config_path} not found, using {fallback_path}")
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_evaluate_config(config_path: str = "evaluate/config.yaml") -> Dict[str, Any]:
    """
    加载评估配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_llm_config(config_path: str = "evaluate/config.yaml") -> Dict[str, Any]:
    """
    获取 LLM 配置

    Returns:
        {
            'host': 'localhost',
            'port': 8000,
            'url': 'http://localhost:8000/generate',
            'model_name': '...',
            'model_path': '...',
            'max_length': 200,
            'temperature': 1.0
        }
    """
    config = load_evaluate_config(config_path)
    llm_config = config['llm']

    host = llm_config['server_host']
    port = llm_config['server_port']

    # 添加 http:// 前缀（如果没有）
    if not host.startswith('http'):
        host = f'http://{host}'

    return {
        'host': llm_config['server_host'],  # 原始 host（不带 http://）
        'port': port,
        'url': f"{host}:{port}/generate",
        'model_name': llm_config.get('model_name', 'unknown'),
        'model_path': llm_config.get('model_path', ''),
        'max_length': llm_config.get('max_length', 200),
        'temperature': llm_config.get('temperature', 1.0)
    }


def get_retriever_config(config_path: str = "evaluate/config.yaml") -> Dict[str, Any]:
    """
    获取 Retriever 配置

    Returns:
        {
            'host': 'localhost',
            'port': 8001,
            'url': 'http://localhost:8001/search',
            'timeout': 30
        }
    """
    config = load_evaluate_config(config_path)
    retriever_config = config['retriever']

    host = retriever_config['host']
    port = retriever_config['port']

    # 添加 http:// 前缀（如果没有）
    if not host.startswith('http'):
        host = f'http://{host}'

    return {
        'host': retriever_config['host'],  # 原始 host
        'port': port,
        'url': f"{host}:{port}/search",
        'timeout': retriever_config.get('timeout', 30)
    }


def get_config_with_overrides(
    config_path: str = "evaluate/config.yaml",
    llm_host: Optional[str] = None,
    llm_port: Optional[int] = None,
    retriever_host: Optional[str] = None,
    retriever_port: Optional[int] = None
) -> Dict[str, Any]:
    """
    获取配置，允许命令行参数覆盖

    Args:
        config_path: 配置文件路径
        llm_host: LLM host（覆盖配置文件）
        llm_port: LLM port（覆盖配置文件）
        retriever_host: Retriever host（覆盖配置文件）
        retriever_port: Retriever port（覆盖配置文件）

    Returns:
        配置字典
    """
    llm_config = get_llm_config(config_path)
    retriever_config = get_retriever_config(config_path)

    # 应用覆盖
    if llm_host is not None:
        llm_config['host'] = llm_host
        if not llm_host.startswith('http'):
            llm_host = f'http://{llm_host}'
        llm_config['url'] = f"{llm_host}:{llm_config['port']}/generate"

    if llm_port is not None:
        llm_config['port'] = llm_port
        host = llm_config['host']
        if not host.startswith('http'):
            host = f'http://{host}'
        llm_config['url'] = f"{host}:{llm_port}/generate"

    if retriever_host is not None:
        retriever_config['host'] = retriever_host
        if not retriever_host.startswith('http'):
            retriever_host = f'http://{retriever_host}'
        retriever_config['url'] = f"{retriever_host}:{retriever_config['port']}/search"

    if retriever_port is not None:
        retriever_config['port'] = retriever_port
        host = retriever_config['host']
        if not host.startswith('http'):
            host = f'http://{host}'
        retriever_config['url'] = f"{host}:{retriever_port}/search"

    return {
        'llm': llm_config,
        'retriever': retriever_config
    }


def get_upper_bound_config(
    module_name: str = None,
    config_path: str = "evaluate/upper_bound_analysis/config.yaml"
) -> Dict[str, Any]:
    """
    获取上限测试模块的完整配置

    Args:
        module_name: 模块名称 (reader_upper_bound, retriever_recall_upperbound, agent_reasoning_check)
        config_path: 配置文件路径

    Returns:
        包含通用配置和模块特定配置的字典
    """
    config = load_upper_bound_config(config_path)

    result = {
        'llm': get_llm_config(config_path),
        'retriever': get_retriever_config(config_path),
        'data': config.get('data', {}),
        'execution': config.get('execution', {})
    }

    # 添加模块特定配置
    if module_name and module_name in config:
        result[module_name] = config[module_name]

    return result


def get_max_samples(config_path: str = "evaluate/upper_bound_analysis/config.yaml") -> Optional[int]:
    """
    获取测试样本数量配置

    Returns:
        max_samples (None 表示使用全部数据)
    """
    config = load_upper_bound_config(config_path)
    return config.get('data', {}).get('max_samples')


def get_datasets(config_path: str = "evaluate/upper_bound_analysis/config.yaml") -> list:
    """
    获取要测试的数据集列表

    Returns:
        数据集名称列表
    """
    config = load_upper_bound_config(config_path)
    return config.get('data', {}).get('datasets', [])


if __name__ == '__main__':
    # 测试配置加载
    print("="*70)
    print("Testing Upper Bound Analysis Config Loading")
    print("="*70)

    try:
        # 测试从统一配置加载
        print("\n1. Loading from upper_bound_analysis/config.yaml:")
        print("-" * 70)

        # LLM 配置
        llm_config = get_llm_config("evaluate/upper_bound_analysis/config.yaml")
        print("\nLLM Config:")
        print(f"  Host: {llm_config['host']}")
        print(f"  Port: {llm_config['port']}")
        print(f"  URL:  {llm_config['url']}")
        print(f"  Model: {llm_config['model_name']}")

        # Retriever 配置
        retriever_config = get_retriever_config("evaluate/upper_bound_analysis/config.yaml")
        print("\nRetriever Config:")
        print(f"  Host: {retriever_config['host']}")
        print(f"  Port: {retriever_config['port']}")
        print(f"  URL:  {retriever_config['url']}")

        # 测试样本数
        max_samples = get_max_samples()
        print(f"\nMax Samples: {max_samples}")

        # 数据集列表
        datasets = get_datasets()
        print(f"Datasets: {', '.join(datasets)}")

        # 测试模块特定配置
        print("\n" + "="*70)
        print("2. Testing module-specific configs:")
        print("-" * 70)

        # Reader Upper Bound
        reader_config = get_upper_bound_config("reader_upper_bound")
        print(f"\nReader Upper Bound:")
        print(f"  Default Prompt Style: {reader_config['reader_upper_bound']['prompt']['default_style']}")
        print(f"  Max Samples: {reader_config['data']['max_samples']}")

        # Retriever Recall Upper Bound
        retriever_ub_config = get_upper_bound_config("retriever_recall_upperbound")
        print(f"\nRetriever Recall Upper Bound:")
        print(f"  K values: {retriever_ub_config['retriever_recall_upperbound']['k_values']}")
        print(f"  Match mode: {retriever_ub_config['retriever_recall_upperbound']['match_mode']}")

        # Agent Reasoning Check
        agent_config = get_upper_bound_config("agent_reasoning_check")
        print(f"\nAgent Reasoning Check:")
        print(f"  Baseline Backend: {agent_config['agent_reasoning_check']['backends']['baseline']['type']}")
        print(f"  Top-K: {agent_config['agent_reasoning_check']['retrieval']['top_k']}")

        print("\n" + "="*70)
        print("✓ All config loading tests passed!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure evaluate/upper_bound_analysis/config.yaml exists")
