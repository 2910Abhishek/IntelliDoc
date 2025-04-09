import chromadb
from chromadb.config import Settings
import os
from src.utils.preprocess import preprocess_text  # Import preprocessing function
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adds IntelliDoc/ to sys.path



def store_in_chroma(input_file, output_dir, collection_name="pdf_chunks"):
    """
    Store preprocessed text chunks and embeddings in Chroma.
    Args:
        input_file (str): Path to extracted text file.
        output_dir (str): Directory for Chroma persistence.
        collection_name (str): Name of the Chroma collection.
    Returns:
        chromadb.Collection: The populated collection.
    """
    # Preprocess text to get chunks with embeddings
    chunks = preprocess_text(input_file, os.path.join(output_dir, "chunks"))
    
    # Initialize Chroma client with persistence
    client = chromadb.PersistentClient(
        path=os.path.join(output_dir, "chroma_db"),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(name=collection_name)
    
    # Prepare data for Chroma
    ids = [chunk['chunk_id'] for chunk in chunks]
    embeddings = [chunk['embedding'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [{'source': input_file, 'chunk_id': chunk['chunk_id']} for chunk in chunks]
    
    # Add to Chroma
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Stored {len(chunks)} chunks in Chroma collection '{collection_name}'")
    return collection

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/Abhishek_extracted.txt"  # From OCR step
    output_dir = "data/processed/"
    collection = store_in_chroma(input_file, output_dir)