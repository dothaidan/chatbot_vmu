from typing import Union
from langchain_chroma import Chroma 
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings 
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

vector_db_path = "src/rag/vector_db_stores/db_chroma"


class VectorDB:
    def __init__(self,
                 documents = None,
                 vector_db: Union[Chroma, FAISS] = Chroma, 
                 embedding = OpenAIEmbeddings(api_key=api_key,
                                              model="text-embedding-3-large")
                 ) -> None:
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents = documents,
                                            embedding = self.embedding,
                                            persist_directory=vector_db_path)
        return db

    def get_retriever(self,
                       search_type: str = "similarity",
                       search_kwargs: dict = {"k": 10}):
        
        retriever = self.db.as_retriever (search_type = search_type,
                                          search_kwargs = search_kwargs)
        return retriever