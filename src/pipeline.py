import os
import pickle
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader


CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
INDEX_PATH = "data/faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()


def ingest(file_path: str) -> int:
    """
    Load, chunk, embed, and store a document.
    Returns number of chunks indexed.
    """
    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    if os.path.exists(INDEX_PATH):
        index = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        index.add_documents(chunks)
    else:
        os.makedirs("data", exist_ok=True)
        index = FAISS.from_documents(chunks, embeddings)

    index.save_local(INDEX_PATH)
    return len(chunks)


def retrieve(query: str, k: int = TOP_K) -> List[str]:
    """
    Retrieve top-k relevant chunks for a query using MMR for diversity.
    """
    if not os.path.exists(INDEX_PATH):
        return []

    index = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = index.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
    return [doc.page_content for doc in docs]
