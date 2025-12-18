#!/usr/bin/env python3
"""
IRCoT 推理链和检索质量分析脚本

分析 2wikimultihopqa 和 musique 数据集在 M 策略下的问题
检查：
1. 推理链长度和质量
2. 检索到的文档相关性
3. 终止条件触发情况
4. 答案提取准确性
"""

import json
import re
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import argparse


class IRCoTAnalyzer:
    """IRCoT 推理链分析器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.stats = defaultdict(lambda: defaultdict(int))
        self.issues = defaultdict(list)

    def load_file(self, filename: str) -> any:
        """加载文件"""
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            return None

        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.readlines()
        return None

    def parse_chain(self, chain_line: str) -> Dict:
        """解析推理链"""
        # 格式: Question\nA: sentence1 A: sentence2 ... S: score
        parts = chain_line.strip().split('\n')
        if len(parts) < 2:
            return {
                'question': chain_line.strip(),
                'reasoning': [],
                'score': None,
                'has_answer': False,
                'answer': None
            }

        question = parts[0]
        reasoning_part = parts[1] if len(parts) > 1 else ""

        # 提取推理句子 (A: xxx A: xxx ...)
        reasoning_sentences = []
        answer_match = None

        # 按 "A:" 分割
        if 'A:' in reasoning_part:
            segments = reasoning_part.split('A:')[1:]  # 跳过第一个空的部分
            for seg in segments:
                # 去掉 S: score 部分
                if 'S:' in seg:
                    seg = seg.split('S:')[0]
                sentence = seg.strip()
                if sentence:
                    reasoning_sentences.append(sentence)

                    # 检查是否包含答案
                    if not answer_match:
                        answer_match = re.search(r'So the answer is:?\s*(.+?)(?:\.|$)', sentence, re.IGNORECASE)
                        if not answer_match:
                            answer_match = re.search(r'Thus.*answer is:?\s*(.+?)(?:\.|$)', sentence, re.IGNORECASE)

        # 提取分数
        score = None
        score_match = re.search(r'S:\s*([\d.]+)', reasoning_part)
        if score_match:
            score = float(score_match.group(1))

        return {
            'question': question,
            'reasoning': reasoning_sentences,
            'score': score,
            'has_answer': answer_match is not None,
            'answer': answer_match.group(1).strip() if answer_match else None,
            'reasoning_length': len(reasoning_sentences)
        }

    def analyze_chain_quality(self, chain: Dict, qid: str) -> Dict:
        """分析单条推理链的质量"""
        issues = []

        # 1. 检查推理链长度
        if chain['reasoning_length'] == 0:
            issues.append("❌ 推理链为空")
            self.stats['empty_chains'] += 1
        elif chain['reasoning_length'] == 1:
            issues.append("⚠️ 推理链只有1句话")
            self.stats['single_sentence_chains'] += 1
        elif chain['reasoning_length'] >= 15:
            issues.append(f"⚠️ 推理链过长 ({chain['reasoning_length']}句)")
            self.stats['too_long_chains'] += 1

        # 2. 检查是否找到答案
        if not chain['has_answer']:
            issues.append("❌ 未找到答案格式 ('So the answer is: ...')")
            self.stats['no_answer_format'] += 1

        # 3. 检查推理质量
        for i, sentence in enumerate(chain['reasoning']):
            # 检查是否包含有效实体（至少有大写字母开头的词）
            has_entity = bool(re.search(r'\b[A-Z][a-z]+', sentence))
            if not has_entity and i < len(chain['reasoning']) - 1:  # 不检查最后一句（可能是答案）
                issues.append(f"⚠️ 第{i+1}句缺少实体: '{sentence[:50]}...'")
                self.stats['sentences_without_entities'] += 1

            # 检查是否是无用的句子
            useless_patterns = [
                r"let'?s think",
                r"we need to find",
                r"we need more information",
                r"based on the context",
                r"according to",
            ]
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in useless_patterns):
                issues.append(f"⚠️ 第{i+1}句可能无用: '{sentence[:50]}...'")
                self.stats['useless_sentences'] += 1

        # 4. 检查是否达到最大迭代次数
        if chain['reasoning_length'] >= 20:
            issues.append("❌ 可能达到最大迭代次数 (20)")
            self.stats['max_iterations_reached'] += 1

        return {
            'qid': qid,
            'issues': issues,
            'reasoning_length': chain['reasoning_length'],
            'has_answer': chain['has_answer'],
            'answer': chain.get('answer')
        }

    def analyze_retrieval_quality(self, contexts: Dict, qid: str) -> Dict:
        """分析检索质量"""
        if qid not in contexts:
            return {'error': 'No contexts found'}

        context_data = contexts[qid]
        issues = []

        # 检查检索到的文档数量
        num_docs = len(context_data) if isinstance(context_data, list) else 1
        self.stats['total_retrieved_docs'] += num_docs

        if num_docs == 0:
            issues.append("❌ 未检索到任何文档")
            self.stats['no_docs_retrieved'] += 1
        elif num_docs < 5:
            issues.append(f"⚠️ 检索到的文档较少 ({num_docs})")
            self.stats['few_docs_retrieved'] += 1

        # TODO: 如果有 ground truth，可以检查相关性

        return {
            'qid': qid,
            'num_docs': num_docs,
            'issues': issues
        }

    def compare_with_gold(self, predictions: Dict, gold_data: Dict) -> Dict:
        """与标准答案对比"""
        correct = 0
        total = 0

        for qid, pred in predictions.items():
            if qid in gold_data:
                total += 1
                gold_answer = gold_data[qid]

                # 简单的字符串匹配
                if self._normalize_answer(pred) == self._normalize_answer(gold_answer):
                    correct += 1
                else:
                    self.issues['wrong_answers'].append({
                        'qid': qid,
                        'predicted': pred,
                        'gold': gold_answer
                    })

        return {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0
        }

    def _normalize_answer(self, text: str) -> str:
        """标准化答案"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def generate_report(self, dataset_name: str) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append(f"IRCoT 分析报告 - {dataset_name}")
        report.append("=" * 80)
        report.append("")

        # 推理链统计
        report.append("## 推理链统计")
        report.append("-" * 80)
        report.append(f"空推理链: {self.stats['empty_chains']}")
        report.append(f"单句推理链: {self.stats['single_sentence_chains']}")
        report.append(f"过长推理链 (≥15句): {self.stats['too_long_chains']}")
        report.append(f"达到最大迭代次数 (20): {self.stats['max_iterations_reached']}")
        report.append(f"未找到答案格式: {self.stats['no_answer_format']}")
        report.append(f"缺少实体的句子: {self.stats['sentences_without_entities']}")
        report.append(f"无用句子: {self.stats['useless_sentences']}")
        report.append("")

        # 检索统计
        report.append("## 检索统计")
        report.append("-" * 80)
        report.append(f"平均检索文档数: {self.stats['total_retrieved_docs'] / max(1, self.stats.get('total_questions', 1)):.2f}")
        report.append(f"未检索到文档: {self.stats['no_docs_retrieved']}")
        report.append(f"检索文档较少 (<5): {self.stats['few_docs_retrieved']}")
        report.append("")

        # 常见问题
        report.append("## 常见问题")
        report.append("-" * 80)

        # 问题分类
        issue_types = defaultdict(int)
        for qid, analysis in self.issues['chain_analysis'].items():
            for issue in analysis['issues']:
                issue_type = issue.split(':')[0].strip()
                issue_types[issue_type] += 1

        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            report.append(f"{issue_type}: {count} 次")

        report.append("")

        # 示例问题
        report.append("## 示例问题 (前5个)")
        report.append("-" * 80)

        count = 0
        for qid, analysis in self.issues['chain_analysis'].items():
            if count >= 5:
                break
            if analysis['issues']:
                count += 1
                report.append(f"\nQID: {qid}")
                report.append(f"推理链长度: {analysis['reasoning_length']}")
                report.append(f"找到答案: {'是' if analysis['has_answer'] else '否'}")
                report.append("问题:")
                for issue in analysis['issues'][:3]:  # 只显示前3个问题
                    report.append(f"  {issue}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def analyze_dataset(self, dataset_name: str,
                       chains_file: str,
                       predictions_file: str,
                       contexts_file: str = None):
        """分析数据集"""
        print(f"\n{'='*80}")
        print(f"开始分析: {dataset_name}")
        print(f"{'='*80}\n")

        # 加载文件
        print("📂 加载文件...")
        chains = self.load_file(chains_file)
        predictions = self.load_file(predictions_file)
        contexts = self.load_file(contexts_file) if contexts_file else None

        if chains is None or predictions is None:
            print("❌ 无法加载必要文件")
            return

        print(f"✓ 加载了 {len(chains)} 条推理链")
        print(f"✓ 加载了 {len(predictions)} 个预测结果")
        if contexts:
            print(f"✓ 加载了 {len(contexts)} 个上下文")

        # 分析每条推理链
        print("\n🔍 分析推理链...")
        self.stats['total_questions'] = len(chains)
        self.issues['chain_analysis'] = {}

        for i, chain_line in enumerate(chains):
            qid = f"{dataset_name}_{i}"

            # 解析推理链
            chain = self.parse_chain(chain_line)

            # 分析质量
            analysis = self.analyze_chain_quality(chain, qid)
            self.issues['chain_analysis'][qid] = analysis

            # 分析检索（如果有）
            if contexts:
                retrieval_analysis = self.analyze_retrieval_quality(contexts, qid)
                analysis['retrieval'] = retrieval_analysis

        # 生成报告
        print("\n📊 生成报告...")
        report = self.generate_report(dataset_name)

        # 保存报告
        report_file = os.path.join(self.output_dir, f"{dataset_name}_analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ 报告已保存到: {report_file}")

        # 打印报告
        print("\n" + report)

        return report


def main():
    parser = argparse.ArgumentParser(description='分析 IRCoT 推理链和检索质量')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录路径 (包含 chains.txt, predictions.json 等文件)')
    parser.add_argument('--dataset', type=str, choices=['2wikimultihopqa', 'musique', 'both'],
                       default='both', help='要分析的数据集')

    args = parser.parse_args()

    datasets = []
    if args.dataset == 'both':
        datasets = ['2wikimultihopqa', 'musique']
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        # 构造文件路径
        # 假设文件命名格式: {dataset}_chains.txt, {dataset}_predictions.json
        analyzer = IRCoTAnalyzer(args.output_dir)

        chains_file = f"{dataset}_chains.txt"
        predictions_file = f"{dataset}_predictions.json"
        contexts_file = f"{dataset}_contexts.json"

        analyzer.analyze_dataset(
            dataset_name=dataset,
            chains_file=chains_file,
            predictions_file=predictions_file,
            contexts_file=contexts_file
        )


if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式输入
    if len(sys.argv) == 1:
        print("=" * 80)
        print("IRCoT 推理链和检索质量分析工具")
        print("=" * 80)
        print()

        output_dir = input("请输入输出目录路径: ").strip()
        if not output_dir:
            print("❌ 输出目录不能为空")
            sys.exit(1)

        if not os.path.exists(output_dir):
            print(f"❌ 目录不存在: {output_dir}")
            sys.exit(1)

        dataset = input("请输入要分析的数据集 (2wikimultihopqa/musique/both) [both]: ").strip() or 'both'

        datasets = []
        if dataset == 'both':
            datasets = ['2wikimultihopqa', 'musique']
        else:
            datasets = [dataset]

        for ds in datasets:
            analyzer = IRCoTAnalyzer(output_dir)

            chains_file = f"{ds}_chains.txt"
            predictions_file = f"{ds}_predictions.json"
            contexts_file = f"{ds}_contexts.json"

            analyzer.analyze_dataset(
                dataset_name=ds,
                chains_file=chains_file,
                predictions_file=predictions_file,
                contexts_file=contexts_file
            )
    else:
        main()
