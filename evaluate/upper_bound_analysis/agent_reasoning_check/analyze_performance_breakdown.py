"""
性能损失分解分析 (Performance Breakdown Analysis)

整合三个测试的结果，生成性能分解图和诊断报告：
1. Reader Upper Bound → 理论上限
2. Retriever Recall Upper Bound → 检索损失
3. Agent Reasoning Check → 推理损失

用于论文的"问题分析"或"实验设置"章节
"""

import json
import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


class PerformanceBreakdownAnalyzer:
    """性能分解分析器"""

    def __init__(self, base_dir: str = "evaluate/upper_bound_analysis"):
        """
        初始化分析器

        Args:
            base_dir: 上限分析框架的基础目录
        """
        self.base_dir = base_dir
        self.reader_dir = os.path.join(base_dir, "reader_upper_bound/outputs")
        self.retriever_dir = os.path.join(base_dir, "retriever_recall_upperbound/outputs")
        self.reasoning_dir = os.path.join(base_dir, "agent_reasoning_check/outputs")

    def load_reader_upper_bound(self, dataset: str, prompt_style: str = "standard") -> float:
        """加载 Reader Upper Bound 的 EM"""
        metrics_file = os.path.join(self.reader_dir, dataset, f"metrics_{prompt_style}.json")

        if not os.path.exists(metrics_file):
            print(f"Warning: Reader metrics not found: {metrics_file}")
            return 0.0

        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return metrics.get('em', 0.0)

    def load_retriever_recall(self, dataset: str) -> Dict[str, float]:
        """加载 Retriever Recall 指标"""
        metrics_file = os.path.join(self.retriever_dir, dataset, "recall_metrics.json")

        if not os.path.exists(metrics_file):
            print(f"Warning: Retriever metrics not found: {metrics_file}")
            return {}

        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return metrics.get('recall_scores', {})

    def load_reasoning_check(self, dataset: str, backend: str = "local_llama") -> float:
        """加载 Reasoning Check 的 EM"""
        backend_slug = backend.replace(' ', '_').replace('/', '_').lower()
        metrics_file = os.path.join(self.reasoning_dir, dataset, f"metrics_{backend_slug}.json")

        if not os.path.exists(metrics_file):
            print(f"Warning: Reasoning metrics not found: {metrics_file}")
            return 0.0

        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return metrics.get('em', 0.0)

    def calculate_breakdown(
        self,
        dataset: str,
        baseline_backend: str = "local_llama",
        strong_backend: str = "gpt4",
        prompt_style: str = "standard"
    ) -> Dict[str, Any]:
        """
        计算性能分解

        Args:
            dataset: 数据集名称
            baseline_backend: 基线 LLM backend
            strong_backend: 强 LLM backend
            prompt_style: Prompt 风格

        Returns:
            性能分解结果
        """
        print(f"\n{'='*70}")
        print(f"Performance Breakdown Analysis: {dataset}")
        print(f"{'='*70}")

        # 1. Reader Upper Bound（理论上限）
        reader_em = self.load_reader_upper_bound(dataset, prompt_style)
        print(f"\n1. Reader Upper Bound (理论上限)")
        print(f"   EM = {reader_em:.4f} ({reader_em*100:.2f}%)")
        print(f"   → 使用 Gold Paragraphs，LLM 能达到的最高性能")

        # 2. Retriever Recall
        retriever_recall = self.load_retriever_recall(dataset)
        recall_5 = retriever_recall.get('recall@5', 0.0)
        recall_100 = retriever_recall.get('recall@100', 0.0)

        print(f"\n2. Retriever Recall Upper Bound (检索上限)")
        print(f"   Recall@100 = {recall_100:.4f} ({recall_100*100:.2f}%)")
        print(f"   Recall@5   = {recall_5:.4f} ({recall_5*100:.2f}%)")
        print(f"   → 检索器能召回正确文档的比例")

        # 3. Baseline RAG Performance
        baseline_em = self.load_reasoning_check(dataset, baseline_backend)
        print(f"\n3. Baseline RAG Performance (基线性能)")
        print(f"   EM = {baseline_em:.4f} ({baseline_em*100:.2f}%)")
        print(f"   → 使用 {baseline_backend}，端到端性能")

        # 4. Strong LLM Performance（如果有的话）
        strong_em = self.load_reasoning_check(dataset, strong_backend)
        if strong_em > 0:
            print(f"\n4. Strong LLM Performance (强 LLM 性能)")
            print(f"   EM = {strong_em:.4f} ({strong_em*100:.2f}%)")
            print(f"   → 使用 {strong_backend}，端到端性能")

        # 计算损失
        print(f"\n{'='*70}")
        print("PERFORMANCE GAPS (性能损失分解)")
        print(f"{'='*70}")

        # 总损失
        total_loss = reader_em - baseline_em
        print(f"\n总损失 (Total Loss):")
        print(f"  {reader_em:.4f} - {baseline_em:.4f} = {total_loss:.4f} ({total_loss*100:.2f}%)")

        # 检索损失（粗略估算）
        # Retrieval Loss ≈ Reader EM × (1 - Recall@5)
        retrieval_loss = reader_em * (1 - recall_5)
        print(f"\n检索损失 (Retrieval Loss):")
        print(f"  ≈ {reader_em:.4f} × (1 - {recall_5:.4f}) = {retrieval_loss:.4f} ({retrieval_loss*100:.2f}%)")
        print(f"  → 由于检索不到正确文档导致的性能下降")

        # 推理损失
        if strong_em > 0:
            reasoning_loss = strong_em - baseline_em
            print(f"\n推理损失 (Reasoning Loss):")
            print(f"  {strong_em:.4f} - {baseline_em:.4f} = {reasoning_loss:.4f} ({reasoning_loss*100:.2f}%)")
            print(f"  → 由于 LLM 逻辑规划能力不足导致的性能下降")
        else:
            reasoning_loss = total_loss - retrieval_loss
            print(f"\n推理损失 (Reasoning Loss, estimated):")
            print(f"  {total_loss:.4f} - {retrieval_loss:.4f} = {reasoning_loss:.4f} ({reasoning_loss*100:.2f}%)")
            print(f"  → 估算的推理损失（未运行强 LLM 测试）")

        # 其他损失
        other_loss = total_loss - retrieval_loss - reasoning_loss
        if abs(other_loss) > 0.01:
            print(f"\n其他损失 (Other Loss):")
            print(f"  {other_loss:.4f} ({other_loss*100:.2f}%)")
            print(f"  → Prompt、格式化等其他因素")

        breakdown = {
            'dataset': dataset,
            'reader_upper_bound': reader_em,
            'retriever_recall_5': recall_5,
            'retriever_recall_100': recall_100,
            'baseline_em': baseline_em,
            'strong_llm_em': strong_em if strong_em > 0 else None,
            'total_loss': total_loss,
            'retrieval_loss': retrieval_loss,
            'reasoning_loss': reasoning_loss,
            'other_loss': other_loss
        }

        # 生成诊断建议
        self._print_diagnosis(breakdown)

        return breakdown

    def _print_diagnosis(self, breakdown: Dict[str, Any]):
        """打印诊断建议"""
        retrieval_loss = breakdown['retrieval_loss']
        reasoning_loss = breakdown['reasoning_loss']
        total_loss = breakdown['total_loss']

        print(f"\n{'='*70}")
        print("DIAGNOSIS & OPTIMIZATION STRATEGY (诊断与优化策略)")
        print(f"{'='*70}")

        # 计算损失占比
        if total_loss > 0:
            retrieval_pct = retrieval_loss / total_loss * 100
            reasoning_pct = reasoning_loss / total_loss * 100
        else:
            retrieval_pct = 0
            reasoning_pct = 0

        print(f"\n性能损失构成：")
        print(f"  - 检索损失: {retrieval_loss:.4f} ({retrieval_pct:.1f}%)")
        print(f"  - 推理损失: {reasoning_loss:.4f} ({reasoning_pct:.1f}%)")

        print(f"\n优化优先级：")

        # 根据损失占比给出建议
        if retrieval_pct > 60:
            print("  1. 【高优先级】优化检索器（检索损失占 {:.1f}%）".format(retrieval_pct))
            print("     - 实现语义门控召回（创新点三）")
            print("     - 使用混合检索（BM25 + Dense）")
            print("     - 优化 Reranker")
            print("  2. 【中优先级】优化推理能力")
            print("     - 使用 CoT Prompting")
        elif reasoning_pct > 60:
            print("  1. 【高优先级】优化推理能力（推理损失占 {:.1f}%）".format(reasoning_pct))
            print("     - 实现 ToT (Tree of Thought) 推理（创新点二）")
            print("     - 使用更强的 LLM")
            print("     - 优化 Prompt")
            print("  2. 【中优先级】优化检索器")
            print("     - 提升召回率")
        else:
            print("  1. 【高优先级】同时优化检索和推理")
            print("     - 检索损失: {:.1f}%".format(retrieval_pct))
            print("     - 推理损失: {:.1f}%".format(reasoning_pct))
            print("  2. 建议策略：")
            print("     - 先实现语义门控召回（提升检索）")
            print("     - 再实现 ToT（提升推理）")

        print(f"\n论文写作建议：")
        print(f"  → 在'问题分析'章节，画出性能分解图")
        print(f"  → 用数据证明创新点的必要性")
        print(f"  → 强调：本文针对性地提出了'自适应召回'和'ToT推理'两个创新点")

        print(f"\n{'='*70}")

    def generate_breakdown_chart(
        self,
        breakdown: Dict[str, Any],
        output_file: str = None
    ):
        """
        生成性能分解图（用于论文）

        Args:
            breakdown: 性能分解数据
            output_file: 输出文件路径
        """
        print(f"\nGenerating performance breakdown chart...")

        # 准备数据
        reader_em = breakdown['reader_upper_bound']
        baseline_em = breakdown['baseline_em']
        retrieval_loss = breakdown['retrieval_loss']
        reasoning_loss = breakdown['reasoning_loss']

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 瀑布图
        categories = [
            'Reader\nUpper Bound\n(理论上限)',
            'Retrieval\nLoss\n(检索损失)',
            'Reasoning\nLoss\n(推理损失)',
            'Baseline\nRAG\n(实际性能)'
        ]

        values = [reader_em, -retrieval_loss, -reasoning_loss, baseline_em]
        colors = ['green', 'red', 'orange', 'blue']

        # 绘制柱状图
        positions = range(len(categories))
        bars = []
        bottom = 0

        for i, (val, color) in enumerate(zip(values, colors)):
            if i == 0:
                # 第一根柱子从0开始
                bar = ax.bar(i, val, color=color, alpha=0.7, edgecolor='black')
                bottom = val
            elif i == len(values) - 1:
                # 最后一根柱子显示最终值
                bar = ax.bar(i, val, color=color, alpha=0.7, edgecolor='black')
            else:
                # 中间的柱子显示下降
                bar = ax.bar(i, -val, bottom=bottom + val, color=color, alpha=0.7, edgecolor='black')
                bottom += val

            bars.append(bar)

            # 添加数值标签
            if i == 0 or i == len(values) - 1:
                label_val = val
            else:
                label_val = val  # 显示负值表示损失

            ax.text(i, bottom + 0.01 if i < len(values) - 1 else val / 2,
                   f'{label_val:.3f}\n({label_val*100:.1f}%)',
                   ha='center', va='bottom' if i < len(values) - 1 else 'center',
                   fontsize=10, fontweight='bold')

        # 连接线
        for i in range(len(categories) - 1):
            if i == 0:
                y_start = reader_em
            else:
                y_start = bottom

            if i == len(categories) - 2:
                y_end = baseline_em
            else:
                y_end = bottom + values[i + 1]

            ax.plot([i + 0.4, i + 0.6], [y_start, y_end], 'k--', linewidth=1, alpha=0.5)

        # 设置标签和标题
        ax.set_ylabel('Exact Match (EM)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Breakdown Analysis - {breakdown["dataset"]}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, reader_em * 1.1)

        # 添加网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # 保存图表
        if output_file is None:
            output_file = os.path.join(
                self.reasoning_dir,
                breakdown['dataset'],
                'performance_breakdown.png'
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Chart saved to {output_file}")

    def generate_comparison_chart(
        self,
        dataset: str,
        backends: List[str],
        output_file: str = None
    ):
        """
        生成不同 LLM 的对比图

        Args:
            dataset: 数据集名称
            backends: Backend 列表
            output_file: 输出文件路径
        """
        print(f"\nGenerating backend comparison chart...")

        # 加载各个 backend 的结果
        em_scores = []
        backend_names = []

        for backend in backends:
            em = self.load_reasoning_check(dataset, backend)
            if em > 0:
                em_scores.append(em)
                backend_names.append(backend)

        if not em_scores:
            print("No data to plot")
            return

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制柱状图
        positions = range(len(backend_names))
        bars = ax.bar(positions, em_scores, color=['blue', 'green', 'orange'][:len(em_scores)],
                     alpha=0.7, edgecolor='black')

        # 添加数值标签
        for i, (bar, em) in enumerate(zip(bars, em_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                   f'{em:.3f}\n({em*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 设置标签和标题
        ax.set_ylabel('Exact Match (EM)', fontsize=12, fontweight='bold')
        ax.set_title(f'LLM Backend Comparison - {dataset}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(backend_names, fontsize=10)
        ax.set_ylim(0, max(em_scores) * 1.15)

        # 添加网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # 保存图表
        if output_file is None:
            output_file = os.path.join(
                self.reasoning_dir,
                dataset,
                'backend_comparison.png'
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Chart saved to {output_file}")

    def generate_report(
        self,
        dataset: str,
        baseline_backend: str = "local_llama",
        strong_backend: str = "gpt4",
        prompt_style: str = "standard",
        output_file: str = None
    ):
        """
        生成完整的性能分解报告

        Args:
            dataset: 数据集名称
            baseline_backend: 基线 backend
            strong_backend: 强 backend
            prompt_style: Prompt 风格
            output_file: 输出文件路径
        """
        # 计算性能分解
        breakdown = self.calculate_breakdown(
            dataset=dataset,
            baseline_backend=baseline_backend,
            strong_backend=strong_backend,
            prompt_style=prompt_style
        )

        # 生成图表
        self.generate_breakdown_chart(breakdown)

        # 如果有多个 backend，生成对比图
        backends = [baseline_backend]
        if breakdown['strong_llm_em']:
            backends.append(strong_backend)

        if len(backends) > 1:
            self.generate_comparison_chart(dataset, backends)

        # 保存报告
        if output_file is None:
            output_file = os.path.join(
                self.reasoning_dir,
                dataset,
                'performance_breakdown_report.md'
            )

        self._save_report(breakdown, output_file)

        print(f"\n✓ Report saved to {output_file}")

    def _save_report(self, breakdown: Dict[str, Any], output_file: str):
        """保存 Markdown 报告"""
        lines = []

        lines.append("# Performance Breakdown Report (性能分解报告)")
        lines.append(f"\nDataset: **{breakdown['dataset']}**")
        lines.append(f"\n## Performance Metrics")
        lines.append(f"\n| Metric | Value | Percentage |")
        lines.append(f"|--------|-------|------------|")
        lines.append(f"| Reader Upper Bound (理论上限) | {breakdown['reader_upper_bound']:.4f} | {breakdown['reader_upper_bound']*100:.2f}% |")
        lines.append(f"| Baseline RAG (实际性能) | {breakdown['baseline_em']:.4f} | {breakdown['baseline_em']*100:.2f}% |")

        if breakdown['strong_llm_em']:
            lines.append(f"| Strong LLM RAG (强LLM性能) | {breakdown['strong_llm_em']:.4f} | {breakdown['strong_llm_em']*100:.2f}% |")

        lines.append(f"\n## Performance Gaps (性能损失)")
        lines.append(f"\n| Loss Type | Value | Percentage |")
        lines.append(f"|-----------|-------|------------|")
        lines.append(f"| Total Loss (总损失) | {breakdown['total_loss']:.4f} | {breakdown['total_loss']*100:.2f}% |")
        lines.append(f"| Retrieval Loss (检索损失) | {breakdown['retrieval_loss']:.4f} | {breakdown['retrieval_loss']*100:.2f}% |")
        lines.append(f"| Reasoning Loss (推理损失) | {breakdown['reasoning_loss']:.4f} | {breakdown['reasoning_loss']*100:.2f}% |")

        lines.append(f"\n## Retrieval Metrics")
        lines.append(f"\n| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Recall@100 | {breakdown['retriever_recall_100']:.4f} ({breakdown['retriever_recall_100']*100:.2f}%) |")
        lines.append(f"| Recall@5 | {breakdown['retriever_recall_5']:.4f} ({breakdown['retriever_recall_5']*100:.2f}%) |")

        lines.append(f"\n## Conclusion")
        lines.append(f"\n本实验通过三个上限测试，系统地分析了 RAG 系统的性能瓶颈：")
        lines.append(f"\n1. **Reader Upper Bound**: 测试 LLM 在获得正确文档时的表现上限")
        lines.append(f"2. **Retriever Recall Upper Bound**: 测试检索器的召回能力")
        lines.append(f"3. **Agent Reasoning Check**: 测试 LLM 的逻辑规划能力")
        lines.append(f"\n结果表明：")

        total_loss = breakdown['total_loss']
        if total_loss > 0:
            retrieval_pct = breakdown['retrieval_loss'] / total_loss * 100
            reasoning_pct = breakdown['reasoning_loss'] / total_loss * 100

            lines.append(f"\n- 检索损失占总损失的 **{retrieval_pct:.1f}%**")
            lines.append(f"- 推理损失占总损失的 **{reasoning_pct:.1f}%**")

        lines.append(f"\n因此，本文针对性地提出了：")
        lines.append(f"\n- **创新点二：ToT (Tree of Thought) 推理** - 解决推理损失")
        lines.append(f"- **创新点三：语义门控自适应召回** - 解决检索损失")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Breakdown Analysis")
    parser.add_argument('--dataset', required=True,
                       help='Dataset name')
    parser.add_argument('--baseline-backend', default='local_llama',
                       help='Baseline LLM backend')
    parser.add_argument('--strong-backend', default='gpt4',
                       help='Strong LLM backend')
    parser.add_argument('--prompt-style', default='standard',
                       help='Prompt style for Reader upper bound')
    parser.add_argument('--base-dir', default='evaluate/upper_bound_analysis',
                       help='Base directory')

    args = parser.parse_args()

    analyzer = PerformanceBreakdownAnalyzer(args.base_dir)

    # 生成报告
    analyzer.generate_report(
        dataset=args.dataset,
        baseline_backend=args.baseline_backend,
        strong_backend=args.strong_backend,
        prompt_style=args.prompt_style
    )


if __name__ == '__main__':
    main()
