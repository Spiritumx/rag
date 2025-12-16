"""
批量更新 prompt 模板文件，扩展到支持 5 个上下文。
"""
import os
from pathlib import Path

# 定义新的模板内容
DIRECT_QA_TEMPLATE = """Please read the following context passages:

---------------------
Wikipedia Title: [Title 1]
[Content 1]

Wikipedia Title: [Title 2]
[Content 2]

Wikipedia Title: [Title 3]
[Content 3]

Wikipedia Title: [Title 4]
[Content 4]

Wikipedia Title: [Title 5]
[Content 5]
---------------------

Based strictly on the context above, answer the following question.

Question: [QUESTION]

Answer:"""

COT_QA_TEMPLATE = """Context information is provided below:

---------------------
Wikipedia Title: [Title 1]
[Content 1]

Wikipedia Title: [Title 2]
[Content 2]

Wikipedia Title: [Title 3]
[Content 3]

Wikipedia Title: [Title 4]
[Content 4]

Wikipedia Title: [Title 5]
[Content 5]
---------------------

Task: Answer the question based on the context. You must think step-by-step.
First, analyze the context to find the relevant facts.
Then, synthesize the facts to form an answer.
Finally, conclude with "So the answer is: [your answer]".

Question: [QUESTION]

Reasoning:"""


def update_templates():
    """批量更新所有 flan_t5 的 context 模板文件"""
    base_dir = Path(__file__).parent / "prompts"

    # 找到所有需要更新的文件
    datasets = ["2wikimultihopqa", "hotpotqa", "iirc", "musique", "nq", "squad", "trivia"]
    distractors = ["1", "2", "3"]

    updated_count = 0

    for dataset in datasets:
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists():
            print(f"跳过不存在的目录: {dataset_dir}")
            continue

        for distractor in distractors:
            # 更新 direct_qa 文件
            direct_qa_file = dataset_dir / f"gold_with_{distractor}_distractors_context_direct_qa_flan_t5.txt"
            if direct_qa_file.exists():
                with open(direct_qa_file, 'w', encoding='utf-8') as f:
                    f.write(DIRECT_QA_TEMPLATE)
                print(f"[OK] Updated: {direct_qa_file.relative_to(base_dir.parent)}")
                updated_count += 1

            # 更新 cot_qa 文件
            cot_qa_file = dataset_dir / f"gold_with_{distractor}_distractors_context_cot_qa_flan_t5.txt"
            if cot_qa_file.exists():
                with open(cot_qa_file, 'w', encoding='utf-8') as f:
                    f.write(COT_QA_TEMPLATE)
                print(f"[OK] Updated: {cot_qa_file.relative_to(base_dir.parent)}")
                updated_count += 1

    print(f"\nTotal updated: {updated_count} files")


if __name__ == "__main__":
    update_templates()
