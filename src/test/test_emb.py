"Test how different embeddings aproaches work."
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from pymilvus import model

doc = Document(id='75fd8474f2c88337f7e0dad69eba0f24ba293cb06693fb' +
                  '746ec403df01a1c0c5',
               content='The Colossus of Rhodes (Ancient Greek: ' +
                       'ὁ Κολοσσὸς Ῥόδιος, ' +
                       'romanized: ho Kolossòs Rhódios Greek: Κολο...',
               meta={'url': 'https://en.wikipedia.org/wiki/Colossus_of_Rhodes',
                     '_split_id': 0})
doc_embedder = SentenceTransformersDocumentEmbedder()
doc_embedder.warm_up()
result = doc_embedder.run([doc])
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',  # Specify the model name
    device='cpu'  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)
# docs = [
#   "Artificial intelligence was founded as an academic discipline in 1956.",
#   "Alan Turing was the first person to conduct substantial research in AI.",
#   "Born in Maida Vale, London, Turing was raised in southern England.",
# ]
docs_embeddings = sentence_transformer_ef.encode_documents([doc.content[0]])
