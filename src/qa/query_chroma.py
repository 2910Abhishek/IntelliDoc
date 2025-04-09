from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_core.runnables import RunnableLambda

def query_chroma(question, db_path="data/processed/chroma_db", collection_name="pdf_chunks"):
    """
    Query Chroma with a dynamic question using a custom QA pipeline.
    Args:
        question (str): User’s question (passed dynamically).
        db_path (str): Path to Chroma database.
        collection_name (str): Name of the Chroma collection.
    Returns:
        str: Polished answer to the question.
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load Chroma vector store
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Initialize Q&A pipeline
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
    
    # Custom retrieval and QA function
    def qa_function(inputs):
        docs = vector_store.similarity_search(inputs["question"], k=3)
        context = '\n\n'.join(doc.page_content for doc in docs)
        if not context:
            return "No relevant information found in the PDF."
        
        answer = qa_pipeline(question=inputs["question"], context=context)
        polished_answer = answer["answer"].strip()
        
        # Post-process for CGPA
        if "CGPA" in inputs["question"].lower():
            import re
            cgpa_match = re.search(r'CGPA\s*(\d+\.\d+)', context)
            if cgpa_match:
                polished_answer = cgpa_match.group(1)
        
        return polished_answer if polished_answer else "No clear answer found."

    # Create a runnable chain
    qa_chain = RunnableLambda(qa_function)
    result = qa_chain.invoke({"question": question})
    
    print(f"Question: {question}\nAnswer: {result}")
    return result

# No hardcoded question in __main__