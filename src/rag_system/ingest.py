"""Import files to build rag algorithm."""
import concurrent.futures
import logging
import multiprocessing
import os

from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

from haystack.utils import ComponentDevice
from milvus_haystack import MilvusDocumentStore
from pymilvus import MilvusException

import torch

import box
from dotenv import load_dotenv, find_dotenv
import yaml

from rag_system.embedders import setup_doc_embedder

load_dotenv(find_dotenv())
logger = logging.getLogger('__main__')
logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import config vars
with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def create_document(doc):
    """Construct a document from loaded file."""
    try:
        if cfg.TYPE_EMB == 'MILVUS_EMB':
            return Document(content=doc["content"])
        if cfg.TYPE_EMB == 'SENT_EMB':
            return Document(content=doc["content"], meta=doc["meta"])
        return None
    except Exception as e:
        logger.error("Error creating document: %s", e)
        raise


def embed_batch_documents(docs_batch, doc_embedder):
    """Perform embedding on the certain document."""
    try:
        if cfg.TYPE_EMB == 'SENT_EMB':
            doc_embedder.warm_up()
            return doc_embedder.run(docs_batch)["documents"]
        if cfg.TYPE_EMB == 'MILVUS_EMB':
            contents = [doc.content for doc in docs_batch]
            encod_docs = doc_embedder.encode_documents(contents)
            for i, doc in enumerate(docs_batch):
                doc.embedding = encod_docs[i]
                docs_batch[i] = doc
            return docs_batch
        return None
    except Exception as e:
        logger.error("Error embedding documents: %s", e)
        raise


def extract_documents():
    """Extract all documents from the dataset in parallel."""
    try:
        dataset = load_dataset('bilgeyucel/seven-wonders', split="train")

        # Check if a GPU is available
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

        device = ComponentDevice.from_str(device_str)
        # Determine the number of CPU cores and threads
        num_cores = multiprocessing.cpu_count()

        # Use up to the number of cores, but not more than dataset size
        num_workers = min(int(num_cores / 2), len(dataset))

        # Parallelize document creation
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers) as executor:
            docs = list(executor.map(create_document, dataset))

        return docs, device, num_workers
    except Exception as e:
        logger.error("Error extracting documents: %s", e)
        raise


def write_documents(doc_store, final_docs, num_workers):
    """Write all documents/their embeddings in the selected document store."""
    # file_paths = glob.glob("./milvus-document-store.md")
    def write_batch(batch):
        try:
            doc_store.write_documents(batch)
        except Exception as e:
            logger.error("Error writing batch to document store: %s", e)
            raise
    # write_batch_size = min(50, len(final_docs), num_workers * 5)

    # Write documents to the store in batches
    # batches = [final_docs[i:i + write_batch_size] for i
    #            in range(0, len(final_docs), write_batch_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) \
         as executor:
        # for batch in batches:
        executor.submit(write_batch, final_docs)

    return doc_store


def convert_documents_into_embeddings():
    """Compute embeddings of the all documents in parallel."""
    docs, device, num_workers = extract_documents()
    doc_embedder = setup_doc_embedder(device=device)
    final_docs = None

    if cfg.TYPE_EMB == 'SENT_EMB':
        # Initialize the embedder once
        doc_embedder.warm_up()
        # Embed documents in batches
        docs_with_embeddings = doc_embedder.run(docs)
        final_docs = docs_with_embeddings["documents"]

    if cfg.TYPE_EMB == 'MILVUS_EMB':
        contents = [doc.content for doc in docs]
        encod_docs = doc_embedder.encode_documents(contents)
        for i, doc in enumerate(docs):
            doc.embedding = encod_docs[i]
            docs[i] = doc
        final_docs = docs

    return final_docs, device, num_workers


def load_text_data_into_inmemory_store() -> InMemoryDocumentStore:
    """Load and embed data into the in-memory document store."""
    # Initialize the document store
    try:
        inmemory_doc_store = InMemoryDocumentStore()
        final_docs, _, num_workers = extract_documents()
        inmemory_doc_store = write_documents(inmemory_doc_store,
                                             final_docs,
                                             num_workers)
        return inmemory_doc_store

    except Exception as e:
        logger.error("Error loading text data into in-memory store: %s", e)
        raise


def load_embedded_data_into_inmemory_store() -> InMemoryDocumentStore:
    """Load and embed data into the in-memory document store."""
    # Initialize the document store
    try:
        inmemory_doc_store = InMemoryDocumentStore()
        final_docs, _, num_workers = convert_documents_into_embeddings()
        inmemory_doc_store = write_documents(inmemory_doc_store,
                                             final_docs,
                                             num_workers)
        return inmemory_doc_store
    except Exception as e:
        logger.error("Error loading embedded data into in-memory store: %s", e)
        raise


def load_embedded_data_into_milvus():
    """Load and embed documents into the milvus doc store/vector database."""
    try:
        milvus_doc_store = MilvusDocumentStore(
            connection_args={
                "host": "milvus-standalone",
                "port": "19530",
                "secure": False,
            },
            drop_old=True,
        )
        logger.info("Connected to Milvus database successfully.")
    except MilvusException as e:
        logger.error("Failed to connect to Milvus database: %s", e)
        return None

    final_docs, _, num_workers = convert_documents_into_embeddings()
    try:
        milvus_doc_store = write_documents(milvus_doc_store,
                                           final_docs,
                                           num_workers)
        logger.info("Documents written to Milvus database successfully.")
    except MilvusException as e:
        logger.error("Failed to write documents to Milvus database: %s", e)

    return milvus_doc_store


def load_data_into_store():
    """Select type of doc store and type of documents/embeddings."""
    doc_store = InMemoryDocumentStore()
    if cfg.TYPE_DOCSTORE == 'inmemory' and cfg.TYPE_RETRIEVAL == 'dense':
        doc_store = load_embedded_data_into_inmemory_store()
    elif cfg.TYPE_DOCSTORE == 'inmemory' and cfg.TYPE_RETRIEVAL == 'sparse':
        doc_store = load_text_data_into_inmemory_store()
    elif cfg.TYPE_DOCSTORE == 'inmemory' and cfg.TYPE_RETRIEVAL == 'hybrid':
        doc_store = load_embedded_data_into_inmemory_store()
    elif cfg.TYPE_DOCSTORE == 'milvus' and cfg.TYPE_RETRIEVAL == 'dense':
        doc_store = load_embedded_data_into_milvus()

    return doc_store


if __name__ == "__main__":
    try:
        load_data_into_store()
    except MilvusException as e:
        logger.error("Failed to load data into store: %s", e)
