#!/usr/bin/env python3
"""
一键运行所有上限测试
Run All Upper Bound Tests

用途：
自动运行三个上限测试模块：
1. Reader Upper Bound (阅读理解上限)
2. Retriever Recall Upper Bound (检索召回上限)
3. Agent Reasoning Check (逻辑推理能力)

使用方法：
    python run_all_tests.py
    python run_all_tests.py --dataset musique --max-samples 50
    python run_all_tests.py --skip-reader --skip-agent
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config_helper import (
        get_upper_bound_config,
        get_max_samples,
        get_datasets,
        get_llm_config,
        get_retriever_config
    )
    CONFIG_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load config_helper: {e}")
    CONFIG_AVAILABLE = False


class UpperBoundTestRunner:
    """上限测试运行器"""

    def __init__(
        self,
        datasets=None,
        max_samples=None,
        parallel_threads=6,
        skip_reader=False,
        skip_retriever=False,
        skip_agent=False,
        predictions_file=None,
        agent_backend="local_llama"
    ):
        """
        初始化测试运行器

        Args:
            datasets: 要测试的数据集列表
            max_samples: 最大样本数
            parallel_threads: 并行线程数
            skip_reader: 跳过 Reader 测试
            skip_retriever: 跳过 Retriever 测试
            skip_agent: 跳过 Agent 测试
            predictions_file: Retriever 测试的预测文件模板
            agent_backend: Agent 测试的 LLM backend
        """
        self.base_dir = Path(__file__).parent
        self.datasets = datasets or ['musique', '2wikimultihopqa']
        self.max_samples = max_samples
        self.parallel_threads = parallel_threads
        self.skip_reader = skip_reader
        self.skip_retriever = skip_retriever
        self.skip_agent = skip_agent
        self.predictions_file = predictions_file
        self.agent_backend = agent_backend

        # 测试结果
        self.results = {
            'reader': {},
            'retriever': {},
            'agent': {}
        }

        # 时间戳
        self.start_time = None
        self.end_time = None

    def print_header(self):
        """打印测试头部信息"""
        print("\n" + "="*80)
        print("UPPER BOUND ANALYSIS - ALL TESTS")
        print("上限分析框架 - 完整测试")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Datasets: {', '.join(self.datasets)}")
        if self.max_samples:
            print(f"Max Samples: {self.max_samples}")
        print(f"Parallel Threads: {self.parallel_threads}")
        print("\nTests to run:")
        print(f"  1. Reader Upper Bound:           {'SKIP' if self.skip_reader else 'YES'}")
        print(f"  2. Retriever Recall Upper Bound: {'SKIP' if self.skip_retriever else 'YES'}")
        print(f"  3. Agent Reasoning Check:        {'SKIP' if self.skip_agent else 'YES'}")
        print("="*80 + "\n")

    def run_reader_upper_bound(self):
        """运行 Reader Upper Bound 测试"""
        if self.skip_reader:
            print("\n[SKIP] Reader Upper Bound Test")
            return

        print("\n" + "="*80)
        print("TEST 1/3: READER UPPER BOUND (阅读理解上限测试)")
        print("="*80)

        script_path = self.base_dir / "reader_upper_bound" / "test_reader_upperbound.py"

        for dataset in self.datasets:
            print(f"\n>>> Testing dataset: {dataset}")

            cmd = [
                sys.executable,
                str(script_path),
                "--datasets", dataset,
                "--parallel-threads", str(self.parallel_threads)
            ]

            if self.max_samples:
                cmd.extend(["--max-samples", str(self.max_samples)])

            try:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
                self.results['reader'][dataset] = "SUCCESS"
                print(f"✓ Reader test completed for {dataset}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Reader test failed for {dataset}: {e}")
                self.results['reader'][dataset] = "FAILED"
            except Exception as e:
                print(f"✗ Error running reader test for {dataset}: {e}")
                self.results['reader'][dataset] = "ERROR"

    def run_retriever_recall(self):
        """运行 Retriever Recall Upper Bound 测试"""
        if self.skip_retriever:
            print("\n[SKIP] Retriever Recall Upper Bound Test")
            return

        print("\n" + "="*80)
        print("TEST 2/3: RETRIEVER RECALL UPPER BOUND (检索召回上限测试)")
        print("="*80)

        script_path = self.base_dir / "retriever_recall_upperbound" / "test_retrieval_recall.py"

        for dataset in self.datasets:
            print(f"\n>>> Testing dataset: {dataset}")

            # 查找预测文件
            if self.predictions_file:
                predictions_file = self.predictions_file.format(dataset=dataset)
            else:
                # 默认路径
                predictions_file = f"outputs/stage2_predictions/{dataset}/predictions_contexts.json"

            # 检查文件是否存在
            if not os.path.exists(predictions_file):
                print(f"⚠ Predictions file not found: {predictions_file}")
                print(f"  Skipping retriever test for {dataset}")
                print(f"  Hint: Run Stage 2 first or specify --predictions-file")
                self.results['retriever'][dataset] = "SKIPPED"
                continue

            cmd = [
                sys.executable,
                str(script_path),
                "--dataset", dataset,
                "--predictions-file", predictions_file,
                "--parallel-threads", str(self.parallel_threads)
            ]

            try:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
                self.results['retriever'][dataset] = "SUCCESS"
                print(f"✓ Retriever test completed for {dataset}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Retriever test failed for {dataset}: {e}")
                self.results['retriever'][dataset] = "FAILED"
            except Exception as e:
                print(f"✗ Error running retriever test for {dataset}: {e}")
                self.results['retriever'][dataset] = "ERROR"

    def run_agent_reasoning_check(self):
        """运行 Agent Reasoning Check 测试"""
        if self.skip_agent:
            print("\n[SKIP] Agent Reasoning Check Test")
            return

        print("\n" + "="*80)
        print("TEST 3/3: AGENT REASONING CHECK (逻辑推理能力测试)")
        print("="*80)

        script_path = self.base_dir / "agent_reasoning_check" / "test_reasoning_ability.py"

        for dataset in self.datasets:
            print(f"\n>>> Testing dataset: {dataset} with backend: {self.agent_backend}")

            cmd = [
                sys.executable,
                str(script_path),
                "--dataset", dataset,
                "--backend", self.agent_backend,
                "--parallel-threads", str(self.parallel_threads)
            ]

            if self.max_samples:
                cmd.extend(["--max-samples", str(self.max_samples)])

            try:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
                self.results['agent'][dataset] = "SUCCESS"
                print(f"✓ Agent test completed for {dataset}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Agent test failed for {dataset}: {e}")
                self.results['agent'][dataset] = "FAILED"
            except Exception as e:
                print(f"✗ Error running agent test for {dataset}: {e}")
                self.results['agent'][dataset] = "ERROR"

    def generate_summary(self):
        """生成测试摘要"""
        print("\n" + "="*80)
        print("TEST SUMMARY (测试摘要)")
        print("="*80)

        # Reader 测试结果
        if not self.skip_reader and self.results['reader']:
            print("\n1. Reader Upper Bound:")
            for dataset, status in self.results['reader'].items():
                icon = "✓" if status == "SUCCESS" else ("⚠" if status == "SKIPPED" else "✗")
                print(f"   {icon} {dataset}: {status}")

        # Retriever 测试结果
        if not self.skip_retriever and self.results['retriever']:
            print("\n2. Retriever Recall Upper Bound:")
            for dataset, status in self.results['retriever'].items():
                icon = "✓" if status == "SUCCESS" else ("⚠" if status == "SKIPPED" else "✗")
                print(f"   {icon} {dataset}: {status}")

        # Agent 测试结果
        if not self.skip_agent and self.results['agent']:
            print("\n3. Agent Reasoning Check:")
            for dataset, status in self.results['agent'].items():
                icon = "✓" if status == "SUCCESS" else ("⚠" if status == "SKIPPED" else "✗")
                print(f"   {icon} {dataset}: {status}")

        # 总体统计
        total_tests = 0
        success_tests = 0
        failed_tests = 0
        skipped_tests = 0

        for module_results in self.results.values():
            for status in module_results.values():
                total_tests += 1
                if status == "SUCCESS":
                    success_tests += 1
                elif status == "FAILED" or status == "ERROR":
                    failed_tests += 1
                elif status == "SKIPPED":
                    skipped_tests += 1

        print("\n" + "-"*80)
        print(f"Total Tests:    {total_tests}")
        print(f"Success:        {success_tests} ✓")
        print(f"Failed/Error:   {failed_tests} ✗")
        print(f"Skipped:        {skipped_tests} ⚠")
        print("-"*80)

        # 时间统计
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            print(f"\nTotal Duration: {duration}")
            print(f"Start Time:     {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End Time:       {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("="*80)

        # 输出文件位置提示
        print("\n📁 Results saved to:")
        print("   - Reader:    evaluate/upper_bound_analysis/reader_upper_bound/outputs/")
        print("   - Retriever: evaluate/upper_bound_analysis/retriever_recall_upperbound/outputs/")
        print("   - Agent:     evaluate/upper_bound_analysis/agent_reasoning_check/outputs/")

        # 下一步建议
        if success_tests == total_tests and total_tests > 0:
            print("\n✓ All tests completed successfully!")
            print("\n📊 Next steps:")
            print("   1. Check diagnosis reports in outputs/ directories")
            print("   2. Run performance breakdown analysis:")
            print("      cd agent_reasoning_check")
            print("      python analyze_performance_breakdown.py --dataset <dataset>")
        elif failed_tests > 0:
            print("\n⚠ Some tests failed. Please check the error messages above.")

        print("\n" + "="*80 + "\n")

    def save_results(self):
        """保存测试结果到文件"""
        output_file = self.base_dir / "outputs" / "all_tests_summary.json"
        output_file.parent.mkdir(exist_ok=True)

        summary = {
            'timestamp': self.start_time.isoformat() if self.start_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            'config': {
                'datasets': self.datasets,
                'max_samples': self.max_samples,
                'parallel_threads': self.parallel_threads
            },
            'results': self.results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✓ Summary saved to {output_file}")

    def run_all(self):
        """运行所有测试"""
        self.start_time = datetime.now()
        self.print_header()

        try:
            # 1. Reader Upper Bound
            self.run_reader_upper_bound()

            # 2. Retriever Recall Upper Bound
            self.run_retriever_recall()

            # 3. Agent Reasoning Check
            self.run_agent_reasoning_check()

        except KeyboardInterrupt:
            print("\n\n⚠ Tests interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\n\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.end_time = datetime.now()

        # 生成摘要
        self.generate_summary()

        # 保存结果
        try:
            self.save_results()
        except Exception as e:
            print(f"Warning: Could not save results: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Run all upper bound tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default config
  python run_all_tests.py

  # Run tests on specific dataset with 50 samples
  python run_all_tests.py --datasets musique --max-samples 50

  # Skip some tests
  python run_all_tests.py --skip-retriever --skip-agent

  # Use custom predictions file
  python run_all_tests.py --predictions-file "path/to/{dataset}/predictions.json"

  # Use different agent backend
  python run_all_tests.py --agent-backend gpt4
        """
    )

    # 数据集和样本
    parser.add_argument('--datasets', nargs='+',
                       default=None,
                       help='Datasets to test (default: from config or [musique, 2wikimultihopqa])')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per dataset (default: from config or 100)')

    # 并行配置
    parser.add_argument('--parallel-threads', type=int, default=6,
                       help='Number of parallel threads (default: 6)')

    # 跳过某些测试
    parser.add_argument('--skip-reader', action='store_true',
                       help='Skip Reader Upper Bound test')
    parser.add_argument('--skip-retriever', action='store_true',
                       help='Skip Retriever Recall Upper Bound test')
    parser.add_argument('--skip-agent', action='store_true',
                       help='Skip Agent Reasoning Check test')

    # Retriever 特定参数
    parser.add_argument('--predictions-file',
                       default=None,
                       help='Predictions file path template (use {dataset} as placeholder)')

    # Agent 特定参数
    parser.add_argument('--agent-backend',
                       default='local_llama',
                       choices=['local_llama', 'gpt4', 'deepseek', 'custom'],
                       help='Agent LLM backend (default: local_llama)')

    args = parser.parse_args()

    # 从配置文件读取默认值（如果可用）
    datasets = args.datasets
    max_samples = args.max_samples

    if CONFIG_AVAILABLE:
        try:
            if not datasets:
                datasets = get_datasets()
            if max_samples is None:
                max_samples = get_max_samples()
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            if not datasets:
                datasets = ['musique', '2wikimultihopqa']
            if max_samples is None:
                max_samples = 100
    else:
        if not datasets:
            datasets = ['musique', '2wikimultihopqa']
        if max_samples is None:
            max_samples = 100

    # 检查是否至少运行一个测试
    if args.skip_reader and args.skip_retriever and args.skip_agent:
        print("Error: Cannot skip all tests. At least one test must be enabled.")
        sys.exit(1)

    # 创建运行器并执行
    runner = UpperBoundTestRunner(
        datasets=datasets,
        max_samples=max_samples,
        parallel_threads=args.parallel_threads,
        skip_reader=args.skip_reader,
        skip_retriever=args.skip_retriever,
        skip_agent=args.skip_agent,
        predictions_file=args.predictions_file,
        agent_backend=args.agent_backend
    )

    runner.run_all()


if __name__ == '__main__':
    main()
