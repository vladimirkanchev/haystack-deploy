"""Import files to build rag algorithm."""

import concurrent.futures
from itertools import chain
import multiprocessing

from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice
import torch

import box
from dotenv import load_dotenv, find_dotenv
import yaml

load_dotenv(find_dotenv())
# Import config vars
with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def create_document(doc):
    """Construct a document from loaded file."""
    return Document(content=doc["content"], meta=doc["meta"])


def embed_documents(docs_batch, model, device):
    """Perform embedding on the certain document."""
    from sentence_transformers import SentenceTransformer

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # model.save('/path/to/local/model')
    doc_embedder = SentenceTransformersDocumentEmbedder(model=model,
                                                        device=device)
    print(doc_embedder)
    doc_embedder.warm_up()
    return doc_embedder.run(docs_batch)["documents"]


def load_data_into_store() -> InMemoryDocumentStore:
    """Load and embed data into the in-memory document store."""
    # Initialize the document store
    document_store = InMemoryDocumentStore()

    # Load the dataset
    dataset = load_dataset('bilgeyucel/seven-wonders', split="train")

    # Check if a GPU is available
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = ComponentDevice.from_str(device_str)
    # Determine the number of CPU cores and threads
    num_cores = multiprocessing.cpu_count()
   
    # Use up to the number of cores, but not more than dataset size
    num_workers = min(num_cores, len(dataset))

    # Parallelize document creation
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers) as executor:
        docs = list(executor.map(create_document, dataset))

    final_docs = None
    if cfg.TYPE_RETRIEVAL in ['dense', 'hybrid']:
        # Initialize the embedder once
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=cfg.EMBEDDINGS, device=device)
        doc_embedder.warm_up()

        # Embed documents in batches
        batch_size = 32 if device == "cuda:0" else 8
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers) as executor:
            futures = [executor.submit(embed_documents, docs[i:i + batch_size],
                                       cfg.EMBEDDINGS, device)
                       for i in range(0, len(docs), batch_size)]
            docs_with_embeddings = [f.result() for f in
                                    concurrent.futures.as_completed(futures)]

        # Flatten the list of lists
        final_docs = list(chain.from_iterable(docs_with_embeddings))

    elif cfg.TYPE_RETRIEVAL == 'sparse':
        final_docs = docs

    if final_docs:
        # Write documents to the store in batches
        # Adjust batch size based on available threads
        write_batch_size = min(50, len(final_docs), num_cores * 5)

        # Write documents to the store in batches
        batches = [final_docs[i:i + write_batch_size] for i
                   in range(0, len(final_docs), write_batch_size)]
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_cores) as executor:
            list(executor.map(document_store.write_documents, batches))

    return document_store
