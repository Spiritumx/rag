import json, jsonlines
import os, sys

def load_json(json_file_path):
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def save_json(json_file_path, json_data):
    if not os.path.exists(os.path.dirname(json_file_path)): 
        os.makedirs(os.path.dirname(json_file_path)) 
    
    with open(json_file_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4, sort_keys=True)
        
    print(json_file_path)

def get_overlapped_qid():
    lst_overlapped_question = []
    with open(os.path.join("raw_data", "musique", 'dev_test_singlehop_questions_v1.0.json')) as input_file:
        single_json_data = json.load(input_file)
    for id_question in single_json_data['natural_questions']:
        lst_overlapped_question.append(id_question['question'])
    return lst_overlapped_question


def get_binary_min_len(lst_dataset_name):
    min_len = float("inf")
    for dataset_name in lst_dataset_name:
        input_file = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd', 'binary', f'{dataset_name}_valid.json')
        json_data = load_json(input_file)
        min_len = len(json_data) if len(json_data) < min_len else min_len
    return min_len

def concat_and_save_binary_silver(binary_input_file, silver_input_file, output_file, min_len = sys.maxsize):
    json_data_binary = load_json(binary_input_file)
    json_data_silver = load_json(silver_input_file)

    lst_silver_ids = [i['id'] for i in json_data_silver]
    json_data_binary = [i for i in json_data_binary if i['id'] not in lst_silver_ids]

    lst_total = json_data_binary + json_data_silver[:min_len]

    save_json(output_file, lst_total)

    print(len(lst_total))

def save_inductive_bias_musique(input_file, output_file, retrieval_mode='bm25', retrieval_type='multi'):
    """
    Save musique data with retrieval labels.
    retrieval_mode: 'bm25', 'hnsw', 'splade', 'hybrid'
    retrieval_type: 'single', 'multi'
    """
    lst_dict_final = []
    
    # Map retrieval mode and type to label
    if retrieval_type == 'single':
        mode_to_label = {
            'bm25': 'S1',      # Single Retriever - BM25
            'hnsw': 'S2',      # Single Retriever - HNSW
            'splade': 'S3',    # Single Retriever - SPLADE
            'hybrid': 'S4'     # Hybrid + Rerank
        }
    else:  # multi
        mode_to_label = {
            'bm25': 'M1',      # Multi-Round BM25
            'hnsw': 'M2',      # Multi-Round HNSW
            'splade': 'M3',    # Multi-Round SPLADE
            'hybrid': 'M4'     # Multi-Round Hybrid + Rerank
        }

    with jsonlines.open(input_file, 'r') as input_file:
        for line in input_file:
            dict_question_complexity = {}
            
            dict_question_complexity['id'] = line['id']
            dict_question_complexity['question'] = line['question']
            dict_question_complexity['answer_description'] = retrieval_type
            dict_question_complexity['answer'] = mode_to_label.get(retrieval_mode, 'M1' if retrieval_type == 'multi' else 'S1')
            dict_question_complexity['dataset_name'] = 'musique'
            dict_question_complexity['retrieval_mode'] = retrieval_mode
            dict_question_complexity['retrieval_type'] = retrieval_type

            lst_dict_final.append(dict_question_complexity)
                    
    save_json(output_file, lst_dict_final)

def save_inductive_bias_single_data(input_file, output_file, dataset_name, set_name, retrieval_mode='bm25'):
    """
    Save single-hop data with single retrieval labels.
    retrieval_mode: 'bm25', 'hnsw', 'splade', 'hybrid'
    """
    json_data = load_json(input_file)

    lst_dict_final = []

    # Map retrieval mode to label for single retrieval
    mode_to_label = {
        'bm25': 'S1',      # Single Retriever - BM25
        'hnsw': 'S2',      # Single Retriever - HNSW
        'splade': 'S3',    # Single Retriever - SPLADE
        'hybrid': 'S4'     # Hybrid + Rerank
    }

    if set_name == 'train' and dataset_name == 'nq':
        lst_overlapped_question = get_overlapped_qid()

    for idx, data in enumerate(json_data):
        if set_name == 'train' and dataset_name == 'nq':
            if data['question'] in lst_overlapped_question:
                    continue

        dict_question_complexity = {}

        dict_question_complexity['id'] = 'single_' + dataset_name + f'_{set_name}_'+str(idx)
        dict_question_complexity['question'] = data['question']
        dict_question_complexity['answer_description'] = 'single'
        dict_question_complexity['answer'] = mode_to_label.get(retrieval_mode, 'S1')
        dict_question_complexity['dataset_name'] = dataset_name
        dict_question_complexity['retrieval_mode'] = retrieval_mode

        lst_dict_final.append(dict_question_complexity)

    save_json(output_file, lst_dict_final)

def save_inductive_bias_hotpotqa(input_file, output_file, retrieval_mode='bm25'):
    """
    Save hotpotqa data with multi-round retrieval labels.
    retrieval_mode: 'bm25', 'hnsw', 'splade', 'hybrid'
    """
    json_data = load_json(input_file)

    lst_dict_final = []

    # Map retrieval mode to label for multi-round retrieval
    mode_to_label = {
        'bm25': 'M1',      # Multi-Round BM25
        'hnsw': 'M2',      # Multi-Round HNSW
        'splade': 'M3',    # Multi-Round SPLADE
        'hybrid': 'M4'     # Multi-Round Hybrid + Rerank
    }

    for idx, data in enumerate(json_data):
        dict_question_complexity = {}

        dict_question_complexity['id'] = data['_id']
        dict_question_complexity['question'] = data['question']
        dict_question_complexity['answer_description'] = 'multi'
        dict_question_complexity['answer'] = mode_to_label.get(retrieval_mode, 'M1')
        dict_question_complexity['dataset_name'] = 'hotpotqa'
        dict_question_complexity['retrieval_mode'] = retrieval_mode

        lst_dict_final.append(dict_question_complexity)

    save_json(output_file, lst_dict_final)



def save_inductive_bias_2wikimultihopqa(input_file, output_file, retrieval_mode='bm25'):
    """
    Save 2wikimultihopqa data with multi-round retrieval labels.
    retrieval_mode: 'bm25', 'hnsw', 'splade', 'hybrid'
    """
    json_data = load_json(input_file)

    lst_dict_final = []

    # Map retrieval mode to label for multi-round retrieval
    mode_to_label = {
        'bm25': 'M1',      # Multi-Round BM25
        'hnsw': 'M2',      # Multi-Round HNSW
        'splade': 'M3',    # Multi-Round SPLADE
        'hybrid': 'M4'     # Multi-Round Hybrid + Rerank
    }

    for idx, data in enumerate(json_data):
        dict_question_complexity = {}

        dict_question_complexity['id'] = data['_id']
        dict_question_complexity['question'] = data['question']
        dict_question_complexity['answer_description'] = 'multi'
        dict_question_complexity['answer'] = mode_to_label.get(retrieval_mode, 'M1')
        dict_question_complexity['dataset_name'] = '2wikimultihopqa'
        dict_question_complexity['retrieval_mode'] = retrieval_mode

        lst_dict_final.append(dict_question_complexity)

    save_json(output_file, lst_dict_final)

def label_complexity(orig_file_path, zero_file_path, one_file_path, multi_file_path, dataset_name, retrieval_mode='bm25'):
    """
    Label complexity based on retrieval strategy.
    retrieval_mode: 'bm25', 'hnsw', 'splade', 'hybrid'
    
    Labels:
    - Z0: Zero-Retrieval (no retrieval)
    - S1-S4: Single retrieval (bm25, hnsw, splade, hybrid)
    - M1-M4: Multi-round retrieval (bm25, hnsw, splade, hybrid)
    """
    # Map retrieval mode to label codes
    single_mode_to_label = {
        'bm25': 'S1',      # Single Retriever - BM25
        'hnsw': 'S2',      # Single Retriever - HNSW
        'splade': 'S3',    # Single Retriever - SPLADE
        'hybrid': 'S4'     # Hybrid + Rerank
    }
    
    multi_mode_to_label = {
        'bm25': 'M1',      # Multi-Round BM25
        'hnsw': 'M2',      # Multi-Round HNSW
        'splade': 'M3',    # Multi-Round SPLADE
        'hybrid': 'M4'     # Multi-Round Hybrid + Rerank
    }
    
    lst_dict_final = []
    with jsonlines.open(orig_file_path, 'r') as input_file:
        for line in input_file:
            dict_question_complexity = {}
            dict_question_complexity['id'] = line['question_id']
            dict_question_complexity['question'] = line['question_text']

            dict_zero = load_json(zero_file_path)
            dict_one = load_json(one_file_path)
            dict_multi = load_json(multi_file_path)

            lst_multi_qid = [i for i in dict_multi.keys()]
            lst_one_qid = [i for i in dict_one.keys()]
            lst_zero_qid = [i for i in dict_zero.keys()]

            if line['question_id'] not in lst_multi_qid + lst_one_qid + lst_zero_qid:
                continue

            dict_question_complexity['dataset_name'] = dataset_name
            dict_question_complexity['retrieval_mode'] = retrieval_mode

            lst_total_answer = []

            if line['question_id'] in lst_multi_qid:
                dict_question_complexity['answer'] = multi_mode_to_label.get(retrieval_mode, 'M1')
                dict_question_complexity['answer_description'] = 'multiple'
                lst_total_answer.append('multiple')
            elif line['question_id'] in lst_one_qid:
                dict_question_complexity['answer'] = single_mode_to_label.get(retrieval_mode, 'S1')
                dict_question_complexity['answer_description'] = 'one'
                lst_total_answer.append('one')
            elif line['question_id'] in lst_zero_qid:
                dict_question_complexity['answer'] = 'Z0'  # Zero-Retrieval
                dict_question_complexity['answer_description'] = 'zero'
                lst_total_answer.append('zero')
            
            dict_question_complexity['total_answer'] = lst_total_answer

            lst_dict_final.append(dict_question_complexity)

    return lst_dict_final

def prepare_predict_file(orig_file_path, dataset_name):
    lst_dict_final = []
    with jsonlines.open(orig_file_path, 'r') as input_file:
        for line in input_file:
            dict_question_doc_count = {}
            dict_question_doc_count['id'] = line['question_id']
            dict_question_doc_count['question'] = line['question_text']

            dict_question_doc_count['dataset_name'] = dataset_name

            lst_total_answer = []
            dict_question_doc_count['answer'] = ''

            dict_question_doc_count['total_answer'] = lst_total_answer

            lst_dict_final.append(dict_question_doc_count)

    return lst_dict_final


def count_stepNum(pred_file):
    dict_qid_to_stepNum = {}
    total_stepNum = 0
    stepNum = 0
    new_qid_flag = False

    with open(pred_file, "r") as f:
        for line in f:
            if line == '\n':
                new_qid_flag = True
                if 'qid' in locals():
                    dict_qid_to_stepNum[qid] = stepNum + 1
                    total_stepNum = total_stepNum + stepNum + 1
                stepNum = 0
                continue

            if new_qid_flag:
                qid = line.strip()
                new_qid_flag = False

            if 'Exit? No.' in line:
                stepNum = stepNum + 1
    
    # last qid
    dict_qid_to_stepNum[qid] = stepNum + 1
    total_stepNum = total_stepNum + stepNum + 1

    output_file = '/'.join(pred_file.split('/')[:-1]) + '/stepNum.json'
    save_json(output_file, dict_qid_to_stepNum)

    print(total_stepNum)


def select_best_strategy_per_question(all_results_by_mode):
    """
    为每个问题选择最优的检索策略
    
    策略选择优先级：
    1. 优先选择能正确回答的策略
    2. 如果多个策略都正确，优先选择更简单的策略（Z0 > S > M）
    3. 如果都不正确，选择 Z0（最简单的策略）
    
    Args:
        all_results_by_mode: dict, key 为检索模式，value 为该模式下的结果列表
        
    Returns:
        list: 每个问题的最优策略标签
    """
    # 首先按问题 ID 组织数据
    questions_by_id = {}
    
    for mode, results in all_results_by_mode.items():
        for item in results:
            qid = item['id']
            if qid not in questions_by_id:
                questions_by_id[qid] = {
                    'question': item['question'],
                    'dataset_name': item['dataset_name'],
                    'total_answer': item['total_answer'],
                    'strategies': {}
                }
            
            # 存储该问题在当前检索模式下的标签
            questions_by_id[qid]['strategies'][mode] = {
                'label': item['answer'],
                'description': item['answer_description'],
                'retrieval_mode': mode
            }
    
    # 为每个问题选择最优策略
    best_results = []
    
    # 策略优先级定义（数字越小优先级越高）
    strategy_priority = {
        'Z0': 0,  # Zero-Retrieval 最简单
        'S1': 1, 'S2': 1, 'S3': 1, 'S4': 1,  # Single Retrieval
        'M1': 2, 'M2': 2, 'M3': 2, 'M4': 2,  # Multi-Round Retrieval 最复杂
    }
    
    for qid, data in questions_by_id.items():
        strategies = data['strategies']
        
        if not strategies:
            continue
        
        # 找出所有正确的策略（answer_description == 'zero' 表示正确）
        correct_strategies = []
        for mode, strategy_info in strategies.items():
            if strategy_info['description'] == 'zero':
                correct_strategies.append((mode, strategy_info))
        
        # 选择最优策略
        if correct_strategies:
            # 如果有正确的策略，选择最简单的（优先级最高的）
            best_mode, best_strategy = min(
                correct_strategies,
                key=lambda x: strategy_priority.get(x[1]['label'], 999)
            )
        else:
            # 如果都不正确，选择最简单的策略（Z0）
            # 找到包含 Z0 标签的模式
            z0_strategy = None
            for mode, strategy_info in strategies.items():
                if strategy_info['label'] == 'Z0':
                    z0_strategy = (mode, strategy_info)
                    break
            
            if z0_strategy:
                best_mode, best_strategy = z0_strategy
            else:
                # 如果没有 Z0，选择优先级最高的
                best_mode, best_strategy = min(
                    strategies.items(),
                    key=lambda x: strategy_priority.get(x[1]['label'], 999)
                )
        
        # 构建结果
        best_results.append({
            'id': qid,
            'question': data['question'],
            'answer': best_strategy['label'],
            'answer_description': best_strategy['description'],
            'dataset_name': data['dataset_name'],
            'retrieval_mode': best_mode,
            'total_answer': data['total_answer']
        })
    
    return best_results

    return dict_qid_to_stepNum