import json
import argparse
import os
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in a JSON file.")
    parser.add_argument("--input_file", type=str, default="data/label_augmented.json", help="Path to the JSON file to analyze")
    args = parser.parse_args()

    # Try to locate the file if it doesn't exist at the exact path
    input_path = args.input_file
    if not os.path.exists(input_path):
        # Try relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, input_path),
            os.path.join(script_dir, "../../", input_path),  # Check project root data dir
            os.path.join(os.getcwd(), input_path)
        ]
        found = False
        for candidate in candidates:
            if os.path.exists(candidate):
                input_path = candidate
                found = True
                break
        
        if not found:
            print(f"Error: File not found: {args.input_file}")
            print(f"Searched in: {args.input_file}, " + ", ".join(candidates))
            return
    else:
        input_path = args.input_file

    print(f"Analyzing file: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    if not isinstance(data, list):
        print("Error: Expected a list of items in the JSON file.")
        return

    total_items = len(data)
    print(f"Total items: {total_items}")

    if total_items == 0:
        print("File is empty.")
        return

    # Fields to check for label
    # augment_with_reasoning.py uses 'recommended_strategy' as the output label, but copies from 'answer'
    # We prioritize 'recommended_strategy' as it is the output field
    label_fields = ['recommended_strategy', 'answer', 'strategy']
    
    # Find which field is being used
    target_field = None
    for field in label_fields:
        if field in data[0]:
            target_field = field
            break
    
    if not target_field:
        print("Could not find a recognized label field (recommended_strategy, answer, strategy).")
        print(f"Available keys in first item: {list(data[0].keys())}")
        return

    print(f"Using label field: '{target_field}'")

    counts = Counter()
    for item in data:
        label = item.get(target_field, "MISSING")
        counts[label] += 1

    print("\nLabel Distribution:")
    print(f"{'Label':<10} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 35)
    
    # Sort by label for consistent output
    for label in sorted(counts.keys()):
        count = counts[label]
        percentage = (count / total_items) * 100
        print(f"{label:<10} | {count:<8} | {percentage:.2f}%")
    
    print("-" * 35)

if __name__ == "__main__":
    main()

