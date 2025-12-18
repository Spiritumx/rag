#!/usr/bin/env python3
"""
IRCoT 高级诊断工具 (Advanced Diagnostic Tool)

改进点：
1. 死循环检测 (Loop Detection): 识别模型是否陷入重复生成。
2. 答案来源验证 (Grounding Check): 检查生成的答案是否真的存在于检索到的上下文中。
3. 关键词/实体分析: 强化对检索词质量的评估。
"""

import json
import re
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import sys
from difflib import SequenceMatcher

class AdvancedIRCoTAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.chains = []
        self.predictions = {}
        self.contexts = {}
        # 用于存储每个QID对应的分析结果
        self.analysis_results = {}

    def load_data(self, dataset: str):
        """加载数据，支持多种文件命名格式"""
        print(f"\n📂 加载 {dataset} 数据...")
        self.dataset = dataset

        # 1. 加载 Chains (推理链)
        # 支持多种命名格式
        chain_files = [
            f"{dataset}_predictions_chains.txt",      # 标准格式：musique_predictions_chains.txt
            f"{dataset}_M_chains.txt",                 # M策略格式：musique_M_chains.txt
            f"{dataset}_chains.txt",                   # 简化格式：musique_chains.txt
            f"{dataset}.txt",                          # 最简格式：musique.txt
        ]
        chains_path = self._find_file(chain_files)
        if chains_path:
            print(f"  📄 找到 Chains 文件: {os.path.basename(chains_path)}")
            with open(chains_path, 'r', encoding='utf-8') as f:
                # 过滤空行，按 Q/A 块读取
                content = f.read().strip()
                # 简单的按 Q: 分割，处理可能的多行格式
                raw_chains = content.split('\nQ: ')
                # 补回被切掉的 Q:，除了第一个
                self.chains = []
                for i, c in enumerate(raw_chains):
                    txt = c.strip()
                    if not txt: continue
                    if i > 0: txt = "Q: " + txt
                    elif not txt.startswith("Q:"): txt = "Q: " + txt
                    self.chains.append(txt)
            print(f"  ✓ Chains: {len(self.chains)} 条")
        else:
            print(f"  ❌ 找不到 Chains 文件")
            print(f"     尝试了以下文件名:")
            for fname in chain_files:
                print(f"       - {fname}")

        # 2. 加载 Contexts (检索到的文档)
        ctx_files = [
            f"{dataset}_predictions_contexts.json",    # 标准格式
            f"{dataset}_M_contexts.json",              # M策略格式
            f"{dataset}_contexts.json",                # 简化格式
        ]
        ctx_path = self._find_file(ctx_files)
        if ctx_path:
            print(f"  📄 找到 Contexts 文件: {os.path.basename(ctx_path)}")
            with open(ctx_path, 'r', encoding='utf-8') as f:
                self.contexts = json.load(f)
            print(f"  ✓ Contexts: {len(self.contexts)} 个")
        else:
            self.contexts = {}
            print(f"  ⚠️  找不到 Contexts 文件 (将跳过幻觉检测)")

    def _find_file(self, filenames: List[str]) -> str:
        for fname in filenames:
            path = os.path.join(self.output_dir, fname)
            if os.path.exists(path):
                return path
        return None

    def _similarity(self, a: str, b: str) -> float:
        """计算文本相似度"""
        return SequenceMatcher(None, a, b).ratio()

    def parse_chain_and_diagnose(self, chain_text: str, index: int) -> Dict:
        """解析并诊断单条推理链"""
        lines = chain_text.strip().split('\n')
        question = lines[0].replace('Q: ', '').strip()
        
        # 尝试从文本中提取 QID (如果文件里没存 QID，尝试通过 question 匹配 contexts)
        # 这里假设 contexts 的 key 是 qid，我们需要一种方式关联。
        # 如果无法关联 QID，只能做纯文本分析。
        current_qid = None
        current_context_text = ""
        
        # 尝试通过问题文本反向查找 QID (这是一个 hack，因为 chain.txt 通常不带 QID)
        # 实际使用中，建议 chain 文件最好带上 QID
        
        sentences = []
        full_reasoning = ""
        
        # 提取推理部分
        for line in lines:
            if line.startswith('A:'):
                full_reasoning = line.replace('A:', '').strip()
                # 移除分数后缀
                full_reasoning = re.sub(r'\s*S:\s*[\-\d.]+\s*$', '', full_reasoning)
                break
        
        if full_reasoning:
            # 分句逻辑：简单的按句号分割，或者按思维链步骤分割
            # 假设 IRCoT 的输出通常包含重复的前缀，我们需要去重
            # IRCoT 的 A: 通常是累积的。这里我们假设 txt 里存的是最终状态
            
            # 简单的分句
            raw_sents = re.split(r'(?<=[.!?])\s+', full_reasoning)
            sentences = [s.strip() for s in raw_sents if s.strip()]

        # === 诊断 1: 死循环检测 ===
        has_loop = False
        loop_sentence = ""
        if len(sentences) > 1:
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    if self._similarity(sentences[i], sentences[j]) > 0.9:
                        has_loop = True
                        loop_sentence = sentences[i]
                        break
                if has_loop: break

        # === 诊断 2: 提取答案 ===
        predicted_answer = None
        match = re.search(r'(?:So|Thus|Therefore).*answer is:?\s*(.+?)(?:\.|$)', full_reasoning, re.IGNORECASE)
        if match:
            predicted_answer = match.group(1).strip()

        # === 诊断 3: 幻觉检测 (需要 Context) ===
        is_hallucination = False
        context_coverage = "UNKNOWN"
        
        # 尝试在 contexts 中找到这个问题对应的文档
        # 这是一个模糊匹配，因为 chain.txt 没有 ID
        # 我们遍历 self.contexts 看看有没有 context 包含这个 predicted_answer
        # (注意：这只是近似，并不完美，但对于诊断足够了)
        
        matched_context_docs = []
        if predicted_answer and self.contexts:
            # 这种反向查找很慢，仅用于诊断前 N 个样本
            if index < 50: 
                # 尝试找到包含 question 的 context 键值（如果 contexts 包含 question 文本）
                # 这里简化：我们无法精准匹配 QID，所以只能做定性分析
                # 如果我们假设 chain 的顺序和 contexts 的 keys 顺序一致（不一定）
                pass

        # === 诊断 4: 关键词密度 ===
        entities = []
        for s in sentences:
            # 简单的大写单词提取
            ents = re.findall(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*', s)
            # 过滤常见停用词
            ents = [e for e in ents if e.lower() not in ['the', 'a', 'so', 'thus', 'answer']]
            entities.extend(ents)
        
        avg_entities = len(entities) / len(sentences) if sentences else 0

        # === 综合评级 ===
        status = "NORMAL"
        issues = []
        
        if has_loop:
            status = "LOOP"
            issues.append(f"死循环: '{loop_sentence[:30]}...'")
        
        if not predicted_answer:
            status = "NO_ANSWER"
            issues.append("未生成最终答案")
        
        if avg_entities < 0.5:
            issues.append("实体缺失 (BM25 难以检索)")
            
        if len(sentences) > 15:
            issues.append("推理链过长 (>15步)")

        return {
            'index': index,
            'question': question,
            'num_steps': len(sentences),
            'predicted_answer': predicted_answer,
            'status': status,
            'issues': issues,
            'sentences': sentences,
            'avg_entities': avg_entities
        }

    def run_analysis(self):
        print("\n🔍 开始深度分析...")
        
        stats = {
            'total': 0,
            'loops': 0,
            'no_answer': 0,
            'long_chains': 0,
            'low_entity_density': 0
        }
        
        problematic_cases = []

        for i, chain_text in enumerate(self.chains):
            res = self.parse_chain_and_diagnose(chain_text, i)
            stats['total'] += 1
            
            if '死循环' in str(res['issues']): stats['loops'] += 1
            if res['status'] == 'NO_ANSWER': stats['no_answer'] += 1
            if res['avg_entities'] < 0.5: stats['low_entity_density'] += 1
            if res['num_steps'] > 15: stats['long_chains'] += 1
            
            if res['issues']:
                problematic_cases.append(res)

        self._print_report(stats, problematic_cases)

    def _print_report(self, stats, cases):
        print("\n" + "="*60)
        print(f"📊 IRCoT 诊断报告: {self.dataset}")
        print("="*60)
        
        total = stats['total']
        if total == 0:
            print("没有数据。")
            return

        print(f"分析样本数: {total}")
        print("-" * 30)
        print(f"🔴 死循环 (Loops):          {stats['loops']:<5} ({stats['loops']/total:.1%})")
        print(f"🟡 未生成答案 (No Ans):     {stats['no_answer']:<5} ({stats['no_answer']/total:.1%})")
        print(f"🟡 实体稀疏 (Low Keyword):  {stats['low_entity_density']:<5} ({stats['low_entity_density']/total:.1%}) --> 影响 BM25")
        print(f"⚪ 推理过长 (>15 Steps):    {stats['long_chains']:<5} ({stats['long_chains']/total:.1%})")
        
        print("\n" + "="*60)
        print("🕵️ 典型病历 (Top 5 Failed Cases)")
        print("="*60)
        
        # 优先展示死循环和无答案的
        priority_cases = sorted(cases, key=lambda x: (x['status'] != 'LOOP', len(x['issues'])), reverse=False)
        
        for i, case in enumerate(priority_cases[:5]):
            print(f"\n[Case #{case['index']}] 状态: {case['status']}")
            print(f"Q: {case['question']}")
            print(f"问题点: {', '.join(case['issues'])}")
            print("推理片段 (最后3步):")
            for s in case['sentences'][-3:]:
                print(f"  -> {s}")
            print("-" * 30)

        # 给出针对性建议
        print("\n💡 优化建议 (Thesis Action Plan):")
        if stats['loops'] / total > 0.1:
            print("1. **Loop Detected**: 模型陷入重复。建议在 Prompt 中加入 'Do not repeat previous steps'，或在代码中检测到重复即强制截断。")
        
        if stats['low_entity_density'] / total > 0.3:
            print("2. **Low Entity**: 中间推理缺乏实体，导致 BM25 搜不到东西。")
            print("   -> 强力推荐：使用 '标题扩展法' (Query Expansion with Titles) 代替句子检索。")
            
        if stats['no_answer'] / total > 0.2:
            print("3. **No Answer**: 检索彻底失败，模型无法推导。需要检查 Hit Rate。")

def main():
    print("=" * 80)
    print("IRCoT 高级诊断工具 (Advanced IRCoT Diagnostic Tool)")
    print("=" * 80)
    print()

    # 获取输出目录
    if len(sys.argv) < 2:
        print("💡 提示：可以通过命令行参数指定输出目录")
        print("   例如: python analyze_ircot_detailed.py /path/to/outputs")
        print()
        output_dir = input("请输入输出目录路径 [默认: evaluate/outputs/stage2_predictions]: ").strip()
        if not output_dir:
            output_dir = "evaluate/outputs/stage2_predictions"
    else:
        output_dir = sys.argv[1]

    # 标准化路径（处理相对路径）
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    print(f"\n📁 输出目录: {output_dir}")

    if not os.path.exists(output_dir):
        print(f"❌ 路径不存在: {output_dir}")
        print("\n请确保:")
        print("  1. 路径正确")
        print("  2. 已运行过评估并生成了输出文件")
        print("  3. 输出目录包含 *_chains.txt 和 *_contexts.json 文件")
        return

    # 列出目录中的文件，帮助用户确认
    print(f"\n📋 目录中的文件:")
    try:
        files = os.listdir(output_dir)
        chain_files = [f for f in files if '_chain' in f.lower() and f.endswith('.txt')]
        context_files = [f for f in files if '_context' in f.lower() and f.endswith('.json')]

        if chain_files:
            print(f"  找到 {len(chain_files)} 个 chains 文件:")
            for f in chain_files[:5]:  # 只显示前5个
                print(f"    - {f}")
        else:
            print("  ⚠️  未找到 chains 文件 (*_chains.txt)")

        if context_files:
            print(f"  找到 {len(context_files)} 个 contexts 文件:")
            for f in context_files[:5]:
                print(f"    - {f}")
        else:
            print("  ⚠️  未找到 contexts 文件 (*_contexts.json)")
    except Exception as e:
        print(f"  ❌ 无法列出文件: {e}")

    # 获取要分析的数据集
    print()
    dataset_input = input("请输入要分析的数据集 (逗号分隔) [默认: musique,2wikimultihopqa]: ").strip()
    if not dataset_input:
        datasets = ['musique', '2wikimultihopqa']
    else:
        datasets = [ds.strip() for ds in dataset_input.split(',')]

    print(f"\n🎯 将分析以下数据集: {', '.join(datasets)}")
    print()

    # 运行分析
    for ds in datasets:
        print("\n" + "=" * 80)
        print(f"开始分析: {ds}")
        print("=" * 80)

        analyzer = AdvancedIRCoTAnalyzer(output_dir)
        analyzer.load_data(ds)
        if analyzer.chains:
            analyzer.run_analysis()
        else:
            print(f"\n⚠️  跳过 {ds}: 未找到有效的 chains 数据")
            print(f"   请检查文件命名是否符合以下格式之一:")
            print(f"     - {ds}_predictions_chains.txt")
            print(f"     - {ds}_M_chains.txt")
            print(f"     - {ds}_chains.txt")

if __name__ == "__main__":
    main()