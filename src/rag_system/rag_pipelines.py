"""Contain wrapper function of separater rag pipeline components."""
from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.document_stores.in_memory import InMemoryDocumentStore

import box
import yaml

from rag_system.llm import setup_single_llm
from rag_system.ingest import load_data_into_store

from rag_system.embedders import setup_embedder
from rag_system.wrapper_prompts import setup_prompt
from rag_system.retrievers import setup_single_retriever
from rag_system.retrievers import setup_hyrbrid_retriever

with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_no_rag_pipeline() -> Pipeline:
    """Build basic no rag pipeline - only request to the llm model."""
    prompt = setup_prompt()
    llm = setup_single_llm(cfg.LLM_MODEL)
    answer_builder = AnswerBuilder()

    no_rag_pipeline = Pipeline()
    no_rag_pipeline.add_component("prompt_builder", prompt)
    no_rag_pipeline.add_component("llm", llm)
    no_rag_pipeline.add_component(instance=answer_builder,
                                  name="answer_builder")

    no_rag_pipeline.connect("prompt_builder", "llm")
    no_rag_pipeline.connect("llm.replies", "answer_builder.replies")
    # no_rag_pipeline.draw(path=cfg.PIPELINE_PATH)

    return no_rag_pipeline


def setup_rag_dense_pipeline() -> Pipeline:
    """Build basic rag haystack pipeline."""
    prompt = setup_prompt()
    doc_store = load_data_into_store()

    llm = setup_single_llm(cfg.LLM_MODEL)
    text_embedder = setup_embedder(cfg.EMBEDDINGS)
    retriever = setup_single_retriever(doc_store)

    dense_pipeline = Pipeline()
    dense_pipeline.add_component("text_embedder", text_embedder)
    dense_pipeline.add_component("retriever", retriever)
    dense_pipeline.add_component("prompt_builder", prompt)
    dense_pipeline.add_component("llm", llm)
    dense_pipeline.add_component(instance=AnswerBuilder(),
                                 name="answer_builder")

    # Now, connect the components to each other
    dense_pipeline.connect("text_embedder.embedding",
                           "retriever.query_embedding")
    dense_pipeline.connect("retriever", "prompt_builder.documents")
    dense_pipeline.connect("prompt_builder", "llm")
    dense_pipeline.connect("llm.replies", "answer_builder.replies")
    dense_pipeline.connect("llm.meta", "answer_builder.meta")
    dense_pipeline.connect("retriever.documents", "answer_builder.documents")
    # dense_pipeline.draw(path=cfg.PIPELINE_PATH)

    return dense_pipeline


def setup_rag_sparse_pipeline() -> Pipeline:
    """Build basic rag haystack pipeline."""
    prompt = setup_prompt()
    llm = setup_single_llm(cfg.LLM_MODEL)
    doc_store = InMemoryDocumentStore()
    bm25_retriever = setup_single_retriever(doc_store)

    sparse_pipeline = Pipeline()
    sparse_pipeline.add_component("retriever", bm25_retriever)
    sparse_pipeline.add_component("prompt_builder", prompt)
    sparse_pipeline.add_component("llm", llm)
    sparse_pipeline.add_component(instance=AnswerBuilder(),
                                  name="answer_builder")
    # Now, connect the components to each other
    sparse_pipeline.connect("retriever", "prompt_builder.documents")
    sparse_pipeline.connect("prompt_builder", "llm")
    sparse_pipeline.connect("llm.replies", "answer_builder.replies")
    sparse_pipeline.connect("llm.meta", "answer_builder.meta")
    sparse_pipeline.connect("retriever.documents", "answer_builder.documents")
    # sparse_pipeline.draw(path=cfg.PIPELINE_PATH)

    return sparse_pipeline


def setup_rag_hybrid_pipeline() -> Pipeline:
    """Build basic rag haystack pipeline."""
    doc_store = load_data_into_store()
    prompt = setup_prompt()
    llm = setup_single_llm(cfg.LLM_MODEL)
    text_embedder = setup_embedder(cfg.EMBEDDINGS)
    embedding_retriever, bm25_retriever = setup_hyrbrid_retriever(doc_store)

    document_joiner = DocumentJoiner()
    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component("text_embedder", text_embedder)
    hybrid_pipeline.add_component("embedding_retriever", embedding_retriever)
    hybrid_pipeline.add_component("bm25_retriever", bm25_retriever)
    hybrid_pipeline.add_component("document_joiner", document_joiner)
    hybrid_pipeline.add_component("ranker", ranker)
    hybrid_pipeline.add_component("prompt_builder", prompt)
    hybrid_pipeline.add_component("llm", llm)
    hybrid_pipeline.add_component(instance=AnswerBuilder(),
                                  name="answer_builder")

    # Now, connect the components to each other
    hybrid_pipeline.connect("text_embedder",
                            "embedding_retriever.query_embedding")
    hybrid_pipeline.connect("bm25_retriever", "document_joiner.documents")
    hybrid_pipeline.connect("embedding_retriever",
                            "document_joiner.documents")
    hybrid_pipeline.connect("document_joiner", "ranker")
    hybrid_pipeline.connect("ranker", "prompt_builder.documents")
    hybrid_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    hybrid_pipeline.connect("llm.replies", "answer_builder.replies")
    hybrid_pipeline.connect("llm.meta", "answer_builder.meta")
    hybrid_pipeline.connect("ranker", "answer_builder.documents")
    # hybrid_pipeline.draw(path=cfg.PIPELINE_PATH)

    return hybrid_pipeline


def select_rag_pipeline() -> Pipeline:
    """Select type of pipeline to load."""
    rag_pipeline = setup_no_rag_pipeline()
    print(cfg.TYPE_RETRIEVAL)
    if cfg.TYPE_RETRIEVAL == 'dense':
        rag_pipeline = setup_rag_dense_pipeline()
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        rag_pipeline = setup_rag_sparse_pipeline()
    elif cfg.TYPE_RETRIEVAL == 'hybrid':
        rag_pipeline = setup_rag_hybrid_pipeline()
    elif cfg.TYPE_RETRIEVAL == 'no_rag':
        rag_pipeline = setup_rag_hybrid_pipeline()

    return rag_pipeline
