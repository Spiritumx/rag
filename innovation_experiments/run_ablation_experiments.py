"""
消融实验脚本 - RAG + Innovations System

实验设计:
- Model A (Ours): 完整系统 (Adaptive + Cascade + MI-RA-ToT)
- Model B: w/o MI-RA-ToT (使用原始 M_core 线性推理)
- Model C: w/o Cascade (移除置信度校验)
- Model D: w/o Adaptive (使用固定权重，端口 8001)

用法:
    python innovation_experiments/run_ablation_experiments.py --datasets squad hotpotqa
    python innovation_experiments/run_ablation_experiments.py --experiments A B C D
    python innovation_experiments/run_ablation_experiments.py --compare-only
"""

import os
import sys
import argparse
import json
import yaml
import copy
from datetime import datetime
from pathlib import Path

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'innovation_experiments'))


# 实验配置定义
EXPERIMENT_CONFIGS = {
    'A': {
        'name': 'Model_A_Full',
        'description': '完整系统 (Adaptive + Cascade + MI-RA-ToT)',
        'config_overrides': {
            'retriever': {'port': 8002},
            'innovations': {
                'adaptive_retrieval': {'enabled': True},
                'cascading_routing': {'enabled': True},
                'tot_reasoning': {'enabled': True},
            }
        }
    },
    'B': {
        'name': 'Model_B_wo_ToT',
        'description': 'w/o MI-RA-ToT (使用线性 M_core)',
        'config_overrides': {
            'retriever': {'port': 8002},
            'innovations': {
                'adaptive_retrieval': {'enabled': True},
                'cascading_routing': {'enabled': True},
                'tot_reasoning': {'enabled': False},  # 禁用 ToT，使用原始 M_core
            }
        }
    },
    'C': {
        'name': 'Model_C_wo_Cascade',
        'description': 'w/o Cascade (移除置信度校验)',
        'config_overrides': {
            'retriever': {'port': 8002},
            'innovations': {
                'adaptive_retrieval': {'enabled': True},
                'cascading_routing': {'enabled': False},  # 禁用级联
                'tot_reasoning': {'enabled': True},
            }
        }
    },
    'D': {
        'name': 'Model_D_wo_Adaptive',
        'description': 'w/o Adaptive (固定权重，端口 8001)',
        'config_overrides': {
            'retriever': {'port': 8001},  # 使用基线检索器
            'innovations': {
                'adaptive_retrieval': {'enabled': False},  # 禁用自适应
                'cascading_routing': {'enabled': True},
                'tot_reasoning': {'enabled': True},
            }
        }
    },
}


def deep_update(base_dict, update_dict):
    """递归更新字典"""
    result = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def create_experiment_config(base_config_path: str, experiment_id: str, output_base: str) -> str:
    """为实验创建配置文件"""
    exp_info = EXPERIMENT_CONFIGS[experiment_id]
    exp_name = exp_info['name']

    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 应用实验特定的覆盖配置
    config = deep_update(config, exp_info['config_overrides'])

    # 修改输出目录为实验特定
    config['outputs']['stage2_dir'] = f'innovation_experiments/ablation_results/{exp_name}/stage2_predictions'
    config['outputs']['stage3_dir'] = f'innovation_experiments/ablation_results/{exp_name}/stage3_metrics'
    config['outputs']['cascade_dir'] = f'innovation_experiments/ablation_results/{exp_name}/cascade_analysis'
    config['execution']['log_file'] = f'innovation_experiments/ablation_results/{exp_name}/pipeline.log'

    # 添加消融标记
    config['ablation'] = {
        'experiment_id': experiment_id,
        'experiment_name': exp_name,
        'description': exp_info['description'],
    }

    # 创建输出目录
    exp_output_dir = os.path.join(output_base, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(exp_output_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  配置文件: {config_path}")
    return config_path


def run_experiment(experiment_id: str, datasets: list, stages: list, base_config_path: str, output_base: str):
    """运行单个消融实验"""
    exp_info = EXPERIMENT_CONFIGS[experiment_id]
    exp_name = exp_info['name']

    print(f"\n{'='*70}")
    print(f"🧪 消融实验: {exp_name}")
    print(f"{'='*70}")
    print(f"描述: {exp_info['description']}")
    print(f"检索器端口: {exp_info['config_overrides']['retriever']['port']}")
    print(f"数据集: {datasets}")
    print(f"阶段: {stages}")
    print(f"{'='*70}")

    # 创建实验配置
    config_path = create_experiment_config(base_config_path, experiment_id, output_base)

    # 运行 V2 Pipeline
    from evaluate_v2.run_pipeline_v2 import PipelineRunnerV2

    runner = PipelineRunnerV2(config_path)
    runner.run(stages=stages, datasets=datasets)

    print(f"\n✓ 实验 {exp_name} 完成")
    print(f"  结果目录: {os.path.join(output_base, exp_name)}")


def compare_results(experiments: list, output_base: str):
    """比较各实验结果"""
    print(f"\n{'='*70}")
    print("📊 消融实验结果对比")
    print(f"{'='*70}")

    all_metrics = {}

    for exp_id in experiments:
        exp_name = EXPERIMENT_CONFIGS[exp_id]['name']
        metrics_path = os.path.join(output_base, exp_name, 'stage3_metrics', 'overall_metrics_v2.json')

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                all_metrics[exp_id] = json.load(f)
        else:
            print(f"  ⚠ 未找到 {exp_name} 的指标文件")

    if not all_metrics:
        print("没有可比较的结果，请先运行实验。")
        return

    # 打印总体对比表
    print("\n📈 总体指标对比:")
    print("-"*80)
    print(f"{'实验':<25} {'EM':<12} {'F1':<12} {'描述':<30}")
    print("-"*80)

    for exp_id in experiments:
        if exp_id in all_metrics:
            m = all_metrics[exp_id]
            exp_info = EXPERIMENT_CONFIGS[exp_id]
            em = m.get('overall', {}).get('em', 0)
            f1 = m.get('overall', {}).get('f1', 0)
            print(f"{exp_info['name']:<25} {em:<12.4f} {f1:<12.4f} {exp_info['description']:<30}")

    print("-"*80)

    # 计算相对于 Model A 的差异
    if 'A' in all_metrics:
        print("\n📉 相对完整系统 (Model A) 的性能变化:")
        print("-"*80)
        print(f"{'实验':<25} {'ΔEM':<12} {'ΔF1':<12} {'验证创新点':<30}")
        print("-"*80)

        baseline = all_metrics['A']
        baseline_em = baseline.get('overall', {}).get('em', 0)
        baseline_f1 = baseline.get('overall', {}).get('f1', 0)

        innovation_map = {
            'B': '创新点3 (MI-RA-ToT)',
            'C': '创新点2 (级联路由)',
            'D': '创新点1 (自适应检索)',
        }

        for exp_id in ['B', 'C', 'D']:
            if exp_id in all_metrics:
                m = all_metrics[exp_id]
                exp_info = EXPERIMENT_CONFIGS[exp_id]
                em = m.get('overall', {}).get('em', 0)
                f1 = m.get('overall', {}).get('f1', 0)
                delta_em = em - baseline_em
                delta_f1 = f1 - baseline_f1
                innovation = innovation_map.get(exp_id, '')
                print(f"{exp_info['name']:<25} {delta_em:+.4f}       {delta_f1:+.4f}       {innovation:<30}")

        print("-"*80)

    # 按数据集对比
    print("\n📊 按数据集对比 EM:")
    datasets = set()
    for m in all_metrics.values():
        # 数据集是顶层键，排除 'overall'
        for key in m.keys():
            if key != 'overall':
                datasets.add(key)

    if datasets:
        print("-"*90)
        header = f"{'实验':<20}"
        for ds in sorted(datasets):
            header += f" {ds:<12}"
        print(header)
        print("-"*90)

        for exp_id in experiments:
            if exp_id in all_metrics:
                m = all_metrics[exp_id]
                exp_name = EXPERIMENT_CONFIGS[exp_id]['name']
                row = f"{exp_name:<20}"
                for ds in sorted(datasets):
                    # 数据集直接是顶层键，其中包含 overall
                    ds_metrics = m.get(ds, {})
                    em = ds_metrics.get('overall', {}).get('em', 0)
                    row += f" {em:<12.4f}"
                print(row)

        print("-"*90)

    # 保存对比报告
    report_path = os.path.join(output_base, 'ablation_comparison.json')
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': experiments,
        'metrics': all_metrics,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n对比报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="运行 RAG 系统消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验设计:
  A - Model A (Ours): 完整系统 (Adaptive + Cascade + MI-RA-ToT)
  B - Model B: w/o MI-RA-ToT (使用原始 M_core 线性推理)
  C - Model C: w/o Cascade (移除置信度校验)
  D - Model D: w/o Adaptive (使用固定权重，端口 8001)

示例:
  # 运行所有消融实验
  python innovation_experiments/run_ablation_experiments.py --datasets squad hotpotqa

  # 运行特定实验
  python innovation_experiments/run_ablation_experiments.py --experiments A B --datasets squad

  # 仅对比已有结果
  python innovation_experiments/run_ablation_experiments.py --compare-only

注意:
  - Model D 需要基线检索器 (端口 8001) 运行
  - Model A/B/C 需要 V2 检索器 (端口 8002) 运行
  - Stage 1 分类结果共享，不会重新运行
        """
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['A', 'B', 'C', 'D'],
        default=['A', 'B', 'C', 'D'],
        help='要运行的实验 (默认: 全部)'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['squad', 'hotpotqa', 'trivia', 'nq', 'musique', '2wikimultihopqa'],
        help='要处理的数据集'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        default=[2, 3],
        help='要运行的阶段 (默认: 2 3)'
    )

    parser.add_argument(
        '--compare-only',
        action='store_true',
        help='仅对比现有结果'
    )

    parser.add_argument(
        '--config',
        default='innovation_experiments/evaluate_v2/config_v2.yaml',
        help='基础配置文件路径'
    )

    args = parser.parse_args()

    output_base = os.path.join(base_dir, 'innovation_experiments', 'ablation_results')
    os.makedirs(output_base, exist_ok=True)

    if args.compare_only:
        compare_results(args.experiments, output_base)
        return

    # 运行实验
    start_time = datetime.now()

    print("\n" + "="*70)
    print("🧪 消融实验 - RAG + Innovations")
    print("="*70)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实验列表: {args.experiments}")
    print(f"数据集: {args.datasets}")
    print("="*70)

    # 打印实验表
    print("\n📋 实验配置:")
    print("-"*70)
    for exp_id in args.experiments:
        exp = EXPERIMENT_CONFIGS[exp_id]
        port = exp['config_overrides']['retriever']['port']
        print(f"  {exp_id}: {exp['name']} (端口 {port})")
        print(f"      {exp['description']}")
    print("-"*70)

    results = {}
    base_config_path = os.path.join(base_dir, args.config)

    for exp_id in args.experiments:
        try:
            run_experiment(exp_id, args.datasets, args.stages, base_config_path, output_base)
            results[exp_id] = 'SUCCESS'
        except Exception as e:
            print(f"\n❌ 实验 {exp_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = f'FAILED: {str(e)}'

    # 总结
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*70}")
    print("✓ 消融实验完成")
    print(f"{'='*70}")
    print(f"总耗时: {duration}")
    print("\n实验状态:")
    for exp_id, status in results.items():
        symbol = "✓" if status == 'SUCCESS' else "❌"
        print(f"  {symbol} {EXPERIMENT_CONFIGS[exp_id]['name']}: {status}")

    # 保存摘要
    summary_path = os.path.join(output_base, 'ablation_summary.json')
    summary = {
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': duration.total_seconds(),
        'experiments': args.experiments,
        'datasets': args.datasets,
        'results': results,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 自动对比结果
    compare_results(args.experiments, output_base)


if __name__ == '__main__':
    main()
