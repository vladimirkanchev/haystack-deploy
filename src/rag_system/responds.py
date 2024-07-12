import json

from typing import Tuple, List

import box
from dotenv import load_dotenv, find_dotenv

from rag_system.eval_pipelines import evaluate_gt_pipeline
from rag_system.inference import run_pipeline
from rag_system.rag_pipelines import select_rag_pipeline

def get_respond_fastapi(query: str) -> Tuple[str, List[str]]:
    """Run inference on the rag pipeline."""
    rag_pipeline = select_rag_pipeline()
    rag_answer, retrieved_docs = run_pipeline(query, rag_pipeline)

    return rag_answer, retrieved_docs

def get_respond_streamlit(query: str) -> Tuple[str, float]:
    """Run inference on the rag pipeline."""
    rag_pipeline = select_rag_pipeline()
    eval_pipeline = evaluate_gt_pipeline()
    rag_answer, retrieved_docs = run_pipeline(query, rag_pipeline)
    responds = eval_pipeline.run({
        "faithfulness": {"questions": [query],
                         "contexts": [retrieved_docs],
                         "predicted_answers": [rag_answer]},
    }
    )

    return rag_answer, responds['faithfulness']['score']
