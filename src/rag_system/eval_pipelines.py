"""Contain evaluation haystack pipelines of the rag algorithm."""
from haystack import Pipeline
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator


def evaluate_gt_pipeline() -> Pipeline:
    """Build basic evaluation haystack pipeline with ground truth data."""
    # model_femb =TextEmbedding(model_name=cfg.EMBEDDINGS)
    # evaluator = SASEvaluator(model=model_femb)
    # evaluator.warm_up()
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    # eval_pipeline.add_component("sas_evaluator",
    #                             evaluator)

    return eval_pipeline
