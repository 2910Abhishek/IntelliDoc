from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.preprocess import preprocess_text
import os
import shutil

def store_in_chroma(input_file, output_dir, collection_name="pdf_chunks"):
    """
    Store preprocessed text chunks and embeddings in Chroma using LangChain.
    Args:
        input_file (str): Path to extracted text file (passed dynamically).
        output_dir (str): Directory for Chroma persistence.
        collection_name (str): Name of the Chroma collection.
    Returns:
        Chroma: The populated vector store.
    """
    # Clear existing Chroma database for fresh context
    chroma_db_path = os.path.join(output_dir, "chroma_db")
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)
    
    # Preprocess text to get chunks
    chunks = preprocess_text(input_file, os.path.join(output_dir, "chunks"))
    texts = [chunk['text'] for chunk in chunks]
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and populate Chroma vector store
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=chroma_db_path,
        collection_name=collection_name
    )
    
    print(f"Stored {len(texts)} chunks in Chroma collection '{collection_name}'")
    return vector_store