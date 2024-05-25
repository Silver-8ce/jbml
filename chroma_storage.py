from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import chromadb
from chromadb.config import Settings



class chroma_storage:

    def __init__(self):        
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None

    def add_embeddings(self, docs):

        self.db = Chroma.from_documents(docs, self.embedding_function)

    def query(self, query: str, size):
        if(self.db is None):
            return None
        docs = self.db.similarity_search(query)
        output = []
        for i in range(size):
            output.append(docs[i])
        return output

