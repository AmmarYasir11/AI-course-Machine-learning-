import os
from typing import List
import pathlib

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama  # if you use Ollama embedding (if supported)

DOCUMENT_FOLDER = r'C:\Users\user\PycharmProjects\PythonProject\data'
FAISS_INDEX_PATH = r'C:\Users\user\PycharmProjects\PythonProject\db\index.faiss'  # folder for index + metadata

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_txt_documents(folder_path: str) -> List[Document]:
    docs = []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder_path, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_or_load_vectorstore(docs: List[Document], embeddings, index_path: str):
    index_file = pathlib.Path(index_path) / "index.faiss"
    if index_file.exists():
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        pathlib.Path(index_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(index_path)
    return vectorstore

def query_vectorstore(vectorstore, query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)

def main():
    docs = load_txt_documents(DOCUMENT_FOLDER)
    print(f"Loaded {len(docs)} documents (files).")
    chunked_docs = chunk_documents(docs)
    print(f"After chunking, we have {len(chunked_docs)} chunks.")

    embeddings = get_embedding_model()

    vectorstore = build_or_load_vectorstore(chunked_docs, embeddings, FAISS_INDEX_PATH)
    print("Vector store ready.")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ('exit', 'quit'):
            break

        relevant_chunks = query_vectorstore(vectorstore, user_input, k=5)

        context = ""
        for doc in relevant_chunks:
            context += f"[SOURCE: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}\n---\n"

        messages = [
            {
                'role': 'system',
                'content': f"You have access to the following documents for context:\n{context}"
            },
            {
                'role': 'user',
                'content': user_input
            }
        ]
        response = ollama.chat(model='llama3.2', messages=messages)
        assistant_message = response['message']['content']
        print("LLaMA:", assistant_message)

if __name__ == "__main__":
    main()
