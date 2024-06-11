import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    if not embeddings:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model="text-embedding-3-large")

    proxy_embeddings = EmbeddingProxy(embeddings)
    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", collection_name))
    db.add_documents(texts)

    return db
