
import logging
from pathlib import Path
from app.core.processing.pdf_processor import PDFProcessor
from app.core.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize processors
    pdf_processor = PDFProcessor()
    rag_pipeline = RAGPipeline()

    # Example usage
    documents_path = Path("data/documents")
    
    if not documents_path.exists():
        logger.info("Creating documents directory...")
        documents_path.mkdir(parents=True, exist_ok=True)
        logger.info("Please place PDF documents in the data/documents directory")
        return

    # Process all PDFs in the documents directory
    processed_documents = []
    for pdf_file in documents_path.glob("*.pdf"):
        logger.info(f"Processing {pdf_file.name}...")
        
        try:
            # Process document
            result = pdf_processor.process_document(str(pdf_file))
            
            if result['status'] == 'success':
                processed_documents.append(result)
                logger.info(f"Successfully processed {pdf_file.name}")
            else:
                logger.error(f"Failed to process {pdf_file.name}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")

    # Add processed documents to RAG pipeline
    if processed_documents:
        logger.info("Adding documents to RAG pipeline...")
        result = rag_pipeline.add_documents(processed_documents)
        if result['status'] == 'success':
            logger.info("Documents added successfully")
            
            # Example query
            question = "What topics are covered in these documents?"
            logger.info(f"Testing RAG pipeline with question: {question}")
            
            response = rag_pipeline.query(question)
            if response['status'] == 'success':
                logger.info("Answer: " + response['answer'])
            else:
                logger.error("Failed to get response: " + response.get('message', 'Unknown error'))
        else:
            logger.error("Failed to add documents: " + result.get('message', 'Unknown error'))

if __name__ == "__main__":
    main()

