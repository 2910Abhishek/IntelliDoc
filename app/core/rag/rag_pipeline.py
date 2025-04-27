import logging
from typing import Dict, List, Optional
from pathlib import Path

from app.database.vector_store.milvus_store import MilvusVectorStore
from app.core.llm.ollama_handler import OllamaHandler
from app.utils.text_chunker import TextChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize RAG pipeline with vector store and LLM components."""
        self.vector_store = MilvusVectorStore.from_config(Path(config_path))
        self.llm = OllamaHandler(model_name="mistral")
        self.text_chunker = TextChunker()
        
    def add_documents(self, documents: List[Dict]) -> Dict:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'content' and 'metadata' fields
        """
        try:
            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self.text_chunker.process_document_chunks(doc['content'])
                all_chunks.extend([chunk['content'] for chunk in chunks])
            
            # Add chunks to vector store
            self.vector_store.add_documents(all_chunks)
            
            return {
                'status': 'success',
                'message': f'Added {len(all_chunks)} chunks to vector store'
            }
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def query(self, 
             question: str, 
             max_results: int = 2,
             temperature: float = 0.7) -> Dict:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User's question
            max_results: Number of relevant chunks to retrieve
            temperature: Temperature for LLM response
        """
        try:
            # Search for relevant chunks
            relevant_chunks = self.vector_store.search(question, k=max_results)
            
            # Combine chunks into context
            context = "\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # Generate response using LLM
            response = self.llm.generate_response(
                prompt=question,
                context=context,
                temperature=temperature
            )
            
            return {
                'status': 'success',
                'answer': response['response'],
                'context': relevant_chunks,
                'model': response['model']
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }