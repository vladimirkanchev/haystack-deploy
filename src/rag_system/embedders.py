"""Contain wrappers of embedder components of tne rag pipeline."""
from typing import Any

import box
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model import DefaultEmbeddingFunction

import yaml


with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_text_embedder():
    """Embed a text accordint to the selected embedded package."""
    if cfg.TYPE_EMB == "SENT_EMB":
        return SentenceTransformersTextEmbedder(model=cfg.EMBD_SENT)
    if cfg.TYPE_EMB == "MILVUS_EMB":
        return DefaultEmbeddingFunction(model_name=cfg.EMBD_MILVUS,
                                        device='cpu')
    return None


def setup_doc_embedder(device: Any):
    """Embed a document according to selected embedded package."""
    if cfg.TYPE_EMB == "SENT_EMB":
        return SentenceTransformersDocumentEmbedder(model=cfg.EMBD_SENT,
                                                    device=device)
    if cfg.TYPE_EMB == "MILVUS_EMB":
        return SentenceTransformerEmbeddingFunction(
            model_name=cfg.EMBD_MILVUS,
            device='cpu')
    return None
