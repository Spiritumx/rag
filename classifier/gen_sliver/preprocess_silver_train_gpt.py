"""
生成银标签数据，为每个问题选择最优的检索策略

该脚本会：
1. 扫描 predictions/dev_500 目录，找到所有可用的检索模式（bm25, hnsw, splade, hybrid）
2. 读取每种检索模式下的预测结果
3. 为每个问题选择最优的策略：
   - 优先选择能正确回答的策略
   - 如果多个策略都正确，选择最简单的（Z0 > S > M）
   - 如果都不正确，选择 Z0（最简单的策略）
4. 生成包含最优策略标签的数据集

使用方法：
    python classifier/gen_sliver/preprocess_silver_train_gpt.py gpt

输出：
    classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/label_best.json
"""

import json
import jsonlines
import os
import re
from preprocess_utils import *
import argparse

def find_available_retrieval_modes(pred_dir, model_name):
    """扫描 predictions 目录，找到所有可用的检索模式"""
    modes = set()
    if not os.path.exists(pred_dir):
        print(f"Warning: Directory {pred_dir} does not exist")
        return []
    
    folders = [f for f in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, f))]
    for folder in folders:
        if f'ircot_qa_{model_name}' in folder or f'oner_qa_{model_name}' in folder:
            match = re.search(r'___retrieval_mode__(\w+)', folder)
            if match:
                modes.add(match.group(1))
    
    return sorted(list(modes))

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name.", choices=("gpt"))
args = parser.parse_args()

# Set your file path accordingly
orig_nq_file = os.path.join("processed_data", "nq", 'dev_500_subsampled.jsonl')
orig_trivia_file = os.path.join("processed_data", "trivia", 'dev_500_subsampled.jsonl')
orig_squad_file = os.path.join("processed_data", "squad", 'dev_500_subsampled.jsonl')
orig_musique_file = os.path.join("processed_data", "musique", 'dev_500_subsampled.jsonl')
orig_hotpotqa_file = os.path.join("processed_data", "hotpotqa", 'dev_500_subsampled.jsonl')
orig_wikimultihopqa_file = os.path.join("processed_data", "2wikimultihopqa", 'dev_500_subsampled.jsonl')

# 扫描所有可用的检索模式
pred_dir = os.path.join("predictions", "dev_500")
available_modes = find_available_retrieval_modes(pred_dir, args.model_name)

if not available_modes:
    print("Warning: No retrieval modes found with '___retrieval_mode__' suffix.")
    print("Falling back to default mode without suffix.")
    available_modes = ['default']
else:
    print(f"Found {len(available_modes)} retrieval modes: {available_modes}")

if len(available_modes) == 1 and available_modes[0] == 'default':
    print("\nNote: Running in compatibility mode for old folder structure.")
    print("For best results, ensure prediction folders have '___retrieval_mode__<mode>' suffix.")

# 为每种检索模式收集数据
all_results_by_mode = {}

for mode in available_modes:
    print(f"\nProcessing retrieval mode: {mode}")
    
    # 构建文件路径
    if mode == 'default':
        # 向后兼容：没有 retrieval_mode 后缀的旧格式
        nq_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__3___distractor_count__1', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json')
        trivia_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__3___distractor_count__1', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
        squad_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__3___distractor_count__1', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
        musique_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__3___distractor_count__1', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
        hotpotqa_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__3___distractor_count__1', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
        wikimultihopqa_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__3___distractor_count__1', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')
        
        nq_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json')
        trivia_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
        squad_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
        musique_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
        hotpotqa_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
        wikimultihopqa_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')
    else:
        # 新格式：带 retrieval_mode 后缀
        nq_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json')
        trivia_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
        squad_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
        musique_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
        hotpotqa_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
        wikimultihopqa_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')
        
        nq_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json')
        trivia_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__6___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
        squad_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
        musique_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
        hotpotqa_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
        wikimultihopqa_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1___retrieval_mode__{mode}', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')
    
    # zero 文件对所有模式都一样
    nq_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_nq____prompt_set_1', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json')
    trivia_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_trivia____prompt_set_1', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
    squad_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_squad____prompt_set_1', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
    musique_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_musique____prompt_set_1', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
    hotpotqa_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_hotpotqa____prompt_set_1', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
    wikimultihopqa_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_2wikimultihopqa____prompt_set_1', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')
    
    # 为当前检索模式生成标签
    try:
        lst_nq = label_complexity(orig_nq_file, nq_zero_file, nq_one_file, nq_multi_file, 'nq', mode)
        lst_trivia = label_complexity(orig_trivia_file, trivia_zero_file, trivia_one_file, trivia_multi_file, 'trivia', mode)
        lst_squad = label_complexity(orig_squad_file, squad_zero_file, squad_one_file, squad_multi_file, 'squad', mode)
        lst_musique = label_complexity(orig_musique_file, musique_zero_file, musique_one_file, musique_multi_file, 'musique', mode)
        lst_hotpotqa = label_complexity(orig_hotpotqa_file, hotpotqa_zero_file, hotpotqa_one_file, hotpotqa_multi_file, 'hotpotqa', mode)
        lst_wikimultihopqa = label_complexity(orig_wikimultihopqa_file, wikimultihopqa_zero_file, wikimultihopqa_one_file, wikimultihopqa_multi_file, '2wikimultihopqa', mode)
        
        # 存储当前模式的结果
        all_results_by_mode[mode] = lst_nq + lst_trivia + lst_squad + lst_musique + lst_hotpotqa + lst_wikimultihopqa
        print(f"  Collected {len(all_results_by_mode[mode])} samples for mode {mode}")
    except FileNotFoundError as e:
        print(f"  Warning: Could not load data for mode {mode}: {e}")
        print(f"  Skipping mode {mode}")
        continue
    except Exception as e:
        print(f"  Error processing mode {mode}: {e}")
        print(f"  Skipping mode {mode}")
        continue

# 检查是否有有效的数据
if not all_results_by_mode:
    print("\nError: No valid data collected from any retrieval mode.")
    print("Please check that prediction files exist in the expected locations.")
    exit(1)

# 选择每个问题的最优策略
print("\n" + "="*60)
print("Selecting best strategy for each question...")
print("="*60)

best_results = select_best_strategy_per_question(all_results_by_mode)

# 保存结果
output_path = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd', args.model_name, 'silver')
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, 'label_best.json')
save_json(output_file, best_results)

print(f"\nGenerated {len(best_results)} labeled samples with best strategies")
print(f"Output saved to: {output_file}")

# 统计每种策略的使用次数
strategy_counts = {}
for item in best_results:
    label = item['answer']
    strategy_counts[label] = strategy_counts.get(label, 0) + 1

print("\nStrategy distribution:")
for label in sorted(strategy_counts.keys()):
    count = strategy_counts[label]
    percentage = (count / len(best_results)) * 100
    print(f"  {label}: {count:4d} ({percentage:5.2f}%)")


