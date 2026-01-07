"""
快速测试脚本

用法：
    python run_quick_test.py                    # 使用默认配置
    python run_quick_test.py --full             # 完整测试（不限制样本数）
    python run_quick_test.py --dataset musique  # 测试指定数据集
"""

import subprocess
import sys
import os


def run_quick_test(dataset='musique', max_samples=100, prompt_style='standard', full=False):
    """
    运行快速测试

    Args:
        dataset: 数据集名称
        max_samples: 最大样本数（None 表示全部）
        prompt_style: Prompt 风格
        full: 是否运行完整测试
    """
    print("="*70)
    print("Reader Upper Bound - Quick Test")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Prompt Style: {prompt_style}")
    print(f"Max Samples: {'All' if full else max_samples}")
    print("="*70)

    # 构建命令
    cmd = [
        sys.executable,
        'test_reader_upperbound.py',
        '--datasets', dataset,
        '--prompt-style', prompt_style
    ]

    if not full:
        cmd.extend(['--max-samples', str(max_samples)])

    # 运行测试
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        print("\n" + "="*70)
        print("Test completed successfully!")
        print("="*70)

        # 询问是否生成分析报告
        print("\nDo you want to generate a diagnosis report? (y/n): ", end='')
        choice = input().strip().lower()

        if choice == 'y':
            print("\nGenerating diagnosis report...")
            analyze_cmd = [
                sys.executable,
                'analyze_results.py',
                '--dataset', dataset,
                '--prompt-style', prompt_style
            ]
            subprocess.run(analyze_cmd, cwd=os.path.dirname(__file__))
    else:
        print("\n" + "="*70)
        print("Test failed!")
        print("="*70)
        sys.exit(1)


def run_prompt_comparison(dataset='musique', max_samples=50):
    """
    运行 Prompt 风格比较

    Args:
        dataset: 数据集名称
        max_samples: 最大样本数
    """
    print("="*70)
    print("Reader Upper Bound - Prompt Comparison")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Max Samples: {max_samples} (per prompt style)")
    print("Testing 3 prompt styles: standard, cot, structured")
    print("="*70)

    prompt_styles = ['standard', 'cot', 'structured']

    for i, style in enumerate(prompt_styles, 1):
        print(f"\n[{i}/3] Testing prompt style: {style}")
        print("-"*70)

        cmd = [
            sys.executable,
            'test_reader_upperbound.py',
            '--datasets', dataset,
            '--prompt-style', style,
            '--max-samples', str(max_samples)
        ]

        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

        if result.returncode != 0:
            print(f"Warning: Test failed for prompt style '{style}'")

    # 生成比较报告
    print("\n" + "="*70)
    print("Generating comparison report...")
    print("="*70)

    analyze_cmd = [
        sys.executable,
        'analyze_results.py',
        '--dataset', dataset,
        '--compare-prompts'
    ]
    subprocess.run(analyze_cmd, cwd=os.path.dirname(__file__))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Quick Test Script for Reader Upper Bound")
    parser.add_argument('--dataset', default='musique',
                       choices=['musique', '2wikimultihopqa', 'hotpotqa', 'squad', 'trivia', 'nq'],
                       help='Dataset to test')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of samples (default: 100)')
    parser.add_argument('--prompt-style', default='standard',
                       choices=['standard', 'cot', 'structured'],
                       help='Prompt style to use')
    parser.add_argument('--full', action='store_true',
                       help='Run full test (no sample limit)')
    parser.add_argument('--compare-prompts', action='store_true',
                       help='Compare different prompt styles')

    args = parser.parse_args()

    if args.compare_prompts:
        # 运行 Prompt 比较
        run_prompt_comparison(
            dataset=args.dataset,
            max_samples=args.max_samples
        )
    else:
        # 运行单次测试
        run_quick_test(
            dataset=args.dataset,
            max_samples=args.max_samples,
            prompt_style=args.prompt_style,
            full=args.full
        )


if __name__ == '__main__':
    main()
