"""
Answer metric -- mostly taken directly from squad_tools of allennlp.
"""
import re
import string
import collections
from typing import Tuple, List
import ftfy

from metrics.metric import Metric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_accuracy(a_gold, a_pred):
    """
    ACC (准确率): 双向判断预测答案和真实答案的包含关系。
    1. 原有逻辑：Gold 在 Pred 里 (解决废话多)
    2. 新增逻辑：Pred 在 Gold 里 (解决 Gold 太长/太细)
    """
    # 标准化后检查包含关系
    normalized_gold = normalize_answer(a_gold)
    normalized_pred = normalize_answer(a_pred)

    # 1. 原有逻辑：检查 Gold 是否在 Pred 中（解决模型废话多的问题）
    if normalized_gold in normalized_pred:
        return 1

    # 2. 新增逻辑：检查 Pred 是否在 Gold 中（解决 Gold 答案太长的问题）
    # 加长度限制，防止预测 "a" 或 "the" 被算对
    if len(normalized_pred) > 3 and normalized_pred in normalized_gold:
        return 1

    # 3. Token 级别的检查（保留原有逻辑）
    gold_tokens = set(get_tokens(a_gold))
    pred_tokens = set(get_tokens(a_pred))

    # 如果真实答案的所有 token 都在预测中，也认为正确
    if gold_tokens and gold_tokens.issubset(pred_tokens):
        return 1

    return 0


def compute_recall(a_gold, a_pred):
    """
    Recall (召回率): 双向判断答案的覆盖情况。
    1. 计算 Gold 的 token 在 Pred 中的覆盖率（原有逻辑）
    2. 如果 Pred 完全包含在 Gold 中，也给高分（新增逻辑）
    """
    # 标准化后的文本
    normalized_gold = normalize_answer(a_gold)
    normalized_pred = normalize_answer(a_pred)

    # 双向字符串包含检查（优先级最高）
    # 1. Gold 在 Pred 里 -> 完美召回
    if normalized_gold in normalized_pred:
        return 1.0

    # 2. Pred 在 Gold 里 -> 也算高召回（加长度限制）
    if len(normalized_pred) > 3 and normalized_pred in normalized_gold:
        return 1.0

    # 3. Token 级别的召回率计算（原有逻辑）
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    if len(gold_toks) == 0:
        # 如果真实答案为空，recall 为 1
        return 1.0

    # 计算真实答案中有多少 token 在预测中
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_covered = sum(common.values())

    recall = num_covered / len(gold_toks)
    return recall


class SquadAnswerEmF1Metric(Metric):
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_acc = 0.0
        self._total_recall = 0.0
        self._count = 0

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):
        #import pdb; pdb.set_trace()
        if isinstance(predicted_answer, list): predicted_answer = predicted_answer[0]
        if isinstance(ground_truth_answers[0], tuple): ground_truth_answers = [i for i in ground_truth_answers[0]]
        #import pdb; pdb.set_trace()
        predicted_answer = ftfy.fix_text(predicted_answer)
        ground_truth_answers = [ftfy.fix_text(e) for e in ground_truth_answers]

        assert isinstance(predicted_answer, str)
        assert isinstance(ground_truth_answers, (Tuple, List))

        exact_scores = metric_max_over_ground_truths(compute_exact, predicted_answer, ground_truth_answers)
        f1_scores = metric_max_over_ground_truths(compute_f1, predicted_answer, ground_truth_answers)
        acc_scores = metric_max_over_ground_truths(compute_accuracy, predicted_answer, ground_truth_answers)
        recall_scores = metric_max_over_ground_truths(compute_recall, predicted_answer, ground_truth_answers)

        self._total_em += int(exact_scores)
        self._total_f1 += f1_scores
        self._total_acc += int(acc_scores)
        self._total_recall += recall_scores
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        accuracy = self._total_acc / self._count if self._count > 0 else 0
        recall = self._total_recall / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {
            "em": round(exact_match, 3),
            "f1": round(f1_score, 3),
            "acc": round(accuracy, 3),
            "recall": round(recall, 3),
            "count": self._count
        }

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_acc = 0.0
        self._total_recall = 0.0
        self._count = 0
