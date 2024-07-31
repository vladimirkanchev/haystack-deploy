"""Contain functions to evaluate rag algorithm."""
from typing import List, Dict, Tuple

from haystack.evaluation.eval_run_result import EvaluationRunResult

import box
import yaml

from rag_system.eval_pipelines import evaluate_gt_pipeline

with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def evaluate_rag(query: List[str],
                 rag_answers: List[str],
                 gt_answers: List[str],
                 retrieved_docs: List[List[str]])\
        -> Tuple[Dict[str, List[str] | List[List[str]]],
                 Dict[str, Dict[str, List[str] | List[List[str]]]]]:
    """Evaluate rag algorithm with ground truth data."""
    eval_pipeline = evaluate_gt_pipeline()
 
    results = eval_pipeline.run({
        "faithfulness": {"questions": query,
                         "contexts":
                         retrieved_docs,
                         "predicted_answers": rag_answers},
        # "sas_evaluator": {"predicted_answers": rag_answers,
        #                  "ground_truth_answers": gt_answers}
    })
    inputs = {
        "question": query,
        "contexts": retrieved_docs,
        "answer": gt_answers,
        "predicted_answer": rag_answers,
    }

    return inputs, results
