import json
import os
import sys
import argparse
from tqdm import tqdm
from preprocess_utils import load_json, save_json

# 添加项目根目录到 sys.path 以便导入 commaqa
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

try:
    from commaqa.models.gpt3generator import GPTGenerator
except ImportError:
    print("Warning: Could not import GPTGenerator. Please ensure commaqa package is in python path.")
    # 定义一个假的 Generator 用于测试（如果没有安装环境）
    class GPTGenerator:
        def __init__(self, **kwargs): pass
        def generate_text_sequence(self, prompt): 
            return json.dumps({
                "complexity_analysis": {"reasoning": "Test reasoning", "noise_risk": "Test risk"}, 
                "search_intent": "Test intent", 
                "rationale": "Test rationale"
            })

def construct_prompt(item):
    question = item['question']
    strategy = item['answer'] # e.g., S3, Z0, M1
    
    # 简单的策略描述映射
    strategy_desc = {
        "Z0": "Zero-Retrieval (LLM directly answers)",
        "S1": "Single BM25 Retrieval",
        "S2": "Single HNSW Retrieval",
        "S3": "Single SPLADE Retrieval",
        "S4": "Single Hybrid Retrieval",
        "M1": "Multi-hop BM25 Retrieval",
        "M2": "Multi-hop HNSW Retrieval",
        "M3": "Multi-hop SPLADE Retrieval",
        "M4": "Multi-hop Hybrid Retrieval"
    }
    desc = strategy_desc.get(strategy, "Unknown Strategy")

    prompt = f"""
You are an expert in Information Retrieval and Question Answering.
Analyze the following question and its optimal retrieval strategy.

Question: "{question}"
Optimal Strategy: {strategy} ({desc})

Please provide a detailed analysis in valid JSON format with the following fields:
1. "complexity_analysis": Object containing:
   - "reasoning": Why this question requires this specific strategy (e.g., factoid, multi-hop, specific entity attributes).
   - "noise_risk": What could go wrong if we used a simpler or different strategy (e.g., keyword matching noise, missing bridge entities).
2. "search_intent": A structured representation of the search intent (e.g., "Entity <relation> Entity").
3. "recommended_strategy": The strategy code (must be "{strategy}").
4. "rationale": A concise summary of why this strategy is best.

Output ONLY the JSON object. Do not include markdown formatting.
"""
    return prompt

import re

def parse_llm_response(response_text, default_strategy):
    try:
        # 尝试清理可能的 markdown 标记
        text = response_text.strip()
        
        # 使用正则表达式提取最外层的 JSON 对象
        # 匹配从第一个 { 到最后一个 } 之间的所有内容
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # 如果没有找到 {}，可能根本不是 JSON
            print(f"No JSON object found in response: {response_text[:50]}...")
            return None
            
        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw text snippet: {response_text[:100]}...")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/label.json")
    parser.add_argument("--output_file", type=str, default="data/label_augmented.json")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use for generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for testing (0 for 10% of data, -1 for all)")
    args = parser.parse_args()

    # 处理相对路径
    if not os.path.isabs(args.input_file):
        # 如果是在 classifier/gen_sliver 目录下运行，需要调整路径
        # 但通常我们假设从根目录运行或者路径是相对于当前工作目录的
        # 这里假设用户会传入正确的相对路径，或者脚本会在项目根目录运行
        pass

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        # 尝试相对于项目根目录查找（如果脚本是在子目录运行）
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        alt_path = os.path.join(project_root, args.input_file)
        if os.path.exists(alt_path):
            print(f"Found file at {alt_path}")
            args.input_file = alt_path
        else:
            return

    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} samples.")
    elif args.limit == 0:
        limit_count = int(len(data) * 0.1)
        if limit_count < 1 and len(data) > 0: limit_count = 1
        data = data[:limit_count]
        print(f"Limiting to 10% of data: {limit_count} samples.")

    print(f"Initializing {args.model}...")
    # 初始化生成器
    # Explicitly set stop=None to avoid passing unsupported parameters to newer models like gpt-5
    generator = GPTGenerator(model=args.model, temperature=0.7, max_tokens=500, stop=None)

    augmented_data = []
    
    print("Starting augmentation...")
    for item in tqdm(data):
        prompt = construct_prompt(item)
        
        # 调用 LLM
        try:
            response_data = generator.generate_text_sequence(prompt)
            
            # 处理 GPTGenerator 返回的 [(text, index)] 格式
            if isinstance(response_data, list) and len(response_data) > 0:
                if isinstance(response_data[0], tuple):
                    response = response_data[0][0]
                else:
                    response = str(response_data[0])
            elif isinstance(response_data, str):
                response = response_data
            else:
                print(f"Unexpected response format for item {item['id']}: {type(response_data)}")
                continue
            
            # 解析响应
            analysis = parse_llm_response(response, item['answer'])
            
            if analysis:
                # 合并原有字段和新生成的字段
                new_item = item.copy()
                new_item.update(analysis)
                # 确保 recommended_strategy 与原始 label 一致
                new_item['recommended_strategy'] = item['answer'] 
                augmented_data.append(new_item)
            else:
                print(f"Skipping item {item['id']} due to parse error.")
                augmented_data.append(item)

        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            augmented_data.append(item)

    print(f"Saving augmented data to {args.output_file}...")
    
    # 确保输出目录存在
    if not os.path.isabs(args.output_file):
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
         # 如果是在项目根目录运行，args.output_file 可能是 "data/label_augmented.json"
         # 如果不是绝对路径，save_json 会处理，但我们需要确保相对于执行位置是正确的
         pass

    save_json(args.output_file, augmented_data)
    print("Done!")

if __name__ == "__main__":
    main()

