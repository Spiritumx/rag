import json
import os
import sys
import argparse
import re
from tqdm import tqdm
from preprocess_utils import load_json, save_json
from openai import OpenAI

# 添加项目根目录到 sys.path 以便导入 preprocess_utils
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/label.json")
    parser.add_argument("--output_file", type=str, default="data/label_augmented.json")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use for generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for testing (0 for 10% of data, -1 for all)")
    args = parser.parse_args()

    # 处理相对路径
    if not os.path.isabs(args.input_file):
        pass

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
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

    print(f"Initializing OpenAI client for model {args.model}...")
    
    # 直接初始化 OpenAI 客户端，绕过 GPTGenerator
    api_key = os.getenv("OPENAI_API_KEY") or "sk-BZQdNZeSwyih3TKpD95fDd83A90e4556A95f7eB7D489C36b"
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.gpt.ge/v1/"
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    augmented_data = []
    
    print("Starting augmentation...")
    for item in tqdm(data):
        prompt = construct_prompt(item)
        
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"} # 强制 JSON 模式
            )
            
            content = response.choices[0].message.content
            
            try:
                analysis = json.loads(content)
                
                # 合并原有字段和新生成的字段
                new_item = item.copy()
                new_item.update(analysis)
                # 确保 recommended_strategy 与原始 label 一致
                new_item['recommended_strategy'] = item['answer'] 
                augmented_data.append(new_item)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON content for item {item['id']}: {e}")
                # 即使解析失败，如果内容不算太坏，也许可以手动修复或记录，这里简单跳过
                print(f"Content was: {content[:100]}...")
                augmented_data.append(item)

        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            augmented_data.append(item)

    print(f"Saving augmented data to {args.output_file}...")
    save_json(args.output_file, augmented_data)
    print("Done!")

if __name__ == "__main__":
    main()
