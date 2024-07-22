"""Import files to build rag algorithm."""

import concurrent.futures
from itertools import chain
import logging
import multiprocessing

from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice
from milvus_haystack import MilvusDocumentStore
import torch

import box
from dotenv import load_dotenv, find_dotenv
import yaml

load_dotenv(find_dotenv())
logger = logging.getLogger('__main__')

# Import config vars
with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def create_document(doc):
    """Construct a document from loaded file."""
    return Document(content=doc["content"], meta=doc["meta"])


def embed_batch_documents(docs_batch, model, device):
    """Perform embedding on the certain document."""
    doc_embedder = SentenceTransformersDocumentEmbedder(model=model,
                                                        device=device)
    doc_embedder.warm_up()
    return doc_embedder.run(docs_batch)["documents"]


def extract_documents():
    """Extract all documents from the dataset in parallel."""
    dataset = load_dataset('bilgeyucel/seven-wonders', split="train")

    # Check if a GPU is available
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = ComponentDevice.from_str(device_str)
    # Determine the number of CPU cores and threads
    num_cores = multiprocessing.cpu_count() - 4

    # Use up to the number of cores, but not more than dataset size
    num_workers = min(num_cores, len(dataset))

    # Parallelize document creation
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers) as executor:
        docs = list(executor.map(create_document, dataset))

    return docs, device, num_workers


def write_documents(doc_store, final_docs, num_workers):
    """Write all documents/their embeddings in the selected document store."""
    # file_paths = glob.glob("./milvus-document-store.md")
    def write_batch(batch):
        try:
            doc_store.write_documents(batch)
        except Exception as e:
            logger.error(f"Error writing batch to document store: {e}")
            raise
    # write_batch_size = min(50, len(final_docs), num_workers * 5)

    # Write documents to the store in batches
    # batches = [final_docs[i:i + write_batch_size] for i
    #            in range(0, len(final_docs), write_batch_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)\
         as executor:
           # for batch in batches:
        executor.submit(write_batch, final_docs)

    
    return doc_store


def convert_documents_into_embeddings():
    """Compute embeddings of the all documents in parallel."""
    docs, device, num_workers = extract_documents()

    # Initialize the embedder once
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=cfg.EMBEDDINGS, device=device)
    doc_embedder.warm_up()
    # Embed documents in batches
    batch_size = 32 if device == "cuda:0" else 8
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers) as executor:
        futures = [executor.submit(embed_batch_documents,
                                   docs[i:i + batch_size],
                                   cfg.EMBEDDINGS, device)
                   for i in range(0, len(docs), batch_size)]
        docs_with_embeddings = [f.result() for f in
                                concurrent.futures.as_completed(futures)]
    
    final_docs = list(chain.from_iterable(docs_with_embeddings))

    return final_docs, device, num_workers


def load_text_data_into_inmemory_store() -> InMemoryDocumentStore:
    """Load and embed data into the in-memory document store."""
    # Initialize the document store
    inmemory_doc_store = InMemoryDocumentStore()
    final_docs, _, num_workers = extract_documents()
    inmemory_doc_store = write_documents(inmemory_doc_store,
                                         final_docs,
                                         num_workers)
    return inmemory_doc_store


def load_embedded_data_into_inmemory_store() -> InMemoryDocumentStore:
    """Load and embed data into the in-memory document store."""
    # Initialize the document store
    inmemory_doc_store = InMemoryDocumentStore()
    final_docs, _, num_workers = convert_documents_into_embeddings()
    inmemory_doc_store = write_documents(inmemory_doc_store,
                                         final_docs,
                                         num_workers)
    return inmemory_doc_store


def load_embedded_data_into_milvus():
    """Load and embed documents into the milvus doc store/vector database."""
    milvus_doc_store = MilvusDocumentStore(
            #sql_url="sqlite:///mydb.db", 
            connection_args={"uri": "./milvus.db"},
            drop_old=True,
    )
    final_docs, _, num_workers = convert_documents_into_embeddings()

    milvus_doc_store = write_documents(milvus_doc_store,
                                       final_docs,
                                       num_workers)
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
