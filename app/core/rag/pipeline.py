from app.core.processing.pdf_processor import PDFProcessor
from app.core.embedding.embedder import Embedder
from app.database.vector_store.milvus_store import MilvusVectorStore  # Updated import path
import ollama
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Orchestrates the RAG pipeline for document processing and querying."""
    
    def __init__(self, config_path: str):
        self.pdf_processor = PDFProcessor()
        self.embedder = Embedder()
        self.vector_store = MilvusVectorStore.from_config(config_path)
        self.llm = ollama.Client()
    
    def process_document(self, uploaded_file) -> Dict:
        """Process PDF and store embeddings in Milvus."""
        try:
            result = self.pdf_processor.process_pdf(uploaded_file)
            if result['status'] == 'success':
                texts = [chunk['text'] for chunk in result['chunks']]
                self.vector_store.add_documents(texts)
                logger.info(f"Processed and stored {uploaded_file.name}")
            return result
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Query the document store and generate an answer."""
        try:
            system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.
            If you cannot find relevant information in the context, say so clearly."""
            
            # Retrieve relevant chunks
            results = self.vector_store.search(question, k=k)
            context = "\n".join([result['text'] for result in results])
            
            # Generate answer
            prompt = f"""Context: {context}

Question: {question}

Please provide a clear and concise answer based on the context provided."""
            response = self.llm.generate(
                model='mistral-nemo',
                prompt=prompt,
                system=system_prompt
            )
            
            return {
                'status': 'success',
                'response': response['response'],
                'sources': results
            }
        except Exception as e:
            logger.error(f"Error querying document: {e}")
            return {'status': 'error', 'error': str(e)}
