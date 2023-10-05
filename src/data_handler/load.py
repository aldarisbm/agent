from typing import List
from langchain.vectorstores import VectorStore

def load_data(index: str, passages: List[str], vector_store: VectorStore):
    vector_store.add_texts(passages)
    print('loaded vectors')
    return