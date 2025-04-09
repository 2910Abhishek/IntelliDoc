import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def query_chroma(question, db_path="data/processed/chroma_db", collection_name="pdf_chunks"):
    """
    Query Chroma with a question and return an answer using a Q&A model.
    Args:
        question (str): User’s question.
        db_path (str): Path to Chroma database.
        collection_name (str): Name of the Chroma collection.
    Returns:
        str: Answer to the question.
    """
    # Initialize Chroma client
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=collection_name)
    
    # Generate embedding for the question
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(question, convert_to_tensor=False).tolist()
    
    # Query Chroma for top 3 relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # Combine retrieved chunks into context
    context = '\n'.join(results['documents'][0])
    if not context:
        return "No relevant information found."
    
    # Use Q&A model to extract answer
    qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    answer = qa_model(question=question, context=context)
    
    print(f"Question: {question}\nAnswer: {answer['answer']}")
    return answer['answer']

if __name__ == "__main__":
    # Example usage
    question = "What is the main topic of the PDF?"
    answer = query_chroma(question)