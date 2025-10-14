import os
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Updated import for embeddings:
from langchain_huggingface import HuggingFaceEmbeddings

import ollama  # your Ollama client

DOCUMENT_FOLDER = r'C:\Users\user\PycharmProjects\PythonProject\data'
FAISS_INDEX_PATH = r'C:\Users\user\PycharmProjects\PythonProject\db\index.faiss'  # folder for index + metadata

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_txt_documents(folder_path: str) -> List[Document]:
    docs = []
    all_files = os.listdir(folder_path)
    print("📁 Files in folder:", all_files)

    for fname in all_files:
        if not fname.lower().endswith('.txt'):
            print(f"❌ Skipping non-.txt file: {fname}")
            continue

        path = os.path.join(folder_path, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"⚠️ Error reading file {fname}: {e}")
            continue

        if not text:
            print(f"⚠️ Skipping empty file: {fname}")
            continue

        print(f"✅ Loaded file: {fname}")
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
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
    return vectorstore


# ✅ This was missing in your file
def query_vectorstore(vectorstore, query: str, k: int = 3):
    """Run similarity search on the vectorstore."""
    return vectorstore.similarity_search(query, k=k)


def append_query_answer(filename: str, query: str, answer: str):
    """Append a query + answer into an existing file."""
    file_path = os.path.join(DOCUMENT_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"❌ File '{filename}' not found in {DOCUMENT_FOLDER}.")
        return

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n[QUERY]: {query}\n[ANSWER]: {answer}\n")
        print(f"✍️ Query & answer saved to {filename}")
    except Exception as e:
        print(f"⚠️ Error writing to {filename}: {e}")


def main():
    docs = load_txt_documents(DOCUMENT_FOLDER)
    print(f"Loaded {len(docs)} documents (files).")
    chunked_docs = chunk_documents(docs)
    print(f"After chunking, we have {len(chunked_docs)} chunks.")

    embeddings = get_embedding_model()
    vectorstore = build_or_load_vectorstore(chunked_docs, embeddings, FAISS_INDEX_PATH)
    print("Vector store ready.")

    # All queries + answers will be saved here
    target_file = "queries.txt"
    target_path = os.path.join(DOCUMENT_FOLDER, target_file)

    # Create file if it doesn’t exist
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("=== Query & Answer Log ===\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ('exit', 'quit'):
            break

        # Search for relevant chunks
        relevant_chunks = query_vectorstore(vectorstore, user_input, k=5)

        context = ""
        for doc in relevant_chunks:
            context += f"[SOURCE: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}\n---\n"

        print("context", context)

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

        # ✅ Save query + answer to the file
        append_query_answer(target_file, user_input, assistant_message)


if __name__ == "__main__":
    main()

