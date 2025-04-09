import re
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

def preprocess_text(input_file, output_dir, chunk_size=500):
    """
    Clean and chunk text from an input file, then generate embeddings.
    Args:
        input_file (str): Path to extracted text file.
        output_dir (str): Directory to save processed chunks.
        chunk_size (int): Max characters per chunk.
    Returns:
        list: List of dictionaries with text, chunk_id, and embedding.
    """
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean text: remove extra spaces, newlines, and special chars
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize spaces
    text = re.sub(r'[^\w\s.,!?]', '', text)   # Remove special chars
    
    # Split into chunks
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append({
                'chunk_id': f"{Path(input_file).stem}_{chunk_id}",
                'text': ' '.join(current_chunk)
            })
            current_chunk = []
            current_length = 0
            chunk_id += 1
    
    if current_chunk:
        chunks.append({
            'chunk_id': f"{Path(input_file).stem}_{chunk_id}",
            'text': ' '.join(current_chunk)
        })
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for chunk in chunks:
        chunk['embedding'] = model.encode(chunk['text'], convert_to_tensor=False).tolist()
    
    # Save chunks to output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, f"{Path(input_file).stem}_chunks.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(f"ID: {chunk['chunk_id']}\nText: {chunk['text']}\n\n")
    
    print(f"Processed {input_file} into {len(chunks)} chunks, saved to {output_file}")
    return chunks

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/Abhishek_extracted.txt"  # From previous step
    output_dir = "data/processed/chunks"
    chunks = preprocess_text(input_file, output_dir)