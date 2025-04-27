
import logging
from pathlib import Path
from app.core.processing.pdf_processor import PDFProcessor
from app.utils.text_chunker import TextChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize processors
    pdf_processor = PDFProcessor()
    text_chunker = TextChunker()

    # Example usage
    documents_path = Path("data/documents")
    
    if not documents_path.exists():
        logger.info("Creating documents directory...")
        documents_path.mkdir(parents=True, exist_ok=True)
        logger.info("Please place PDF documents in the data/documents directory")
        return

    # Process all PDFs in the documents directory
    for pdf_file in documents_path.glob("*.pdf"):
        logger.info(f"Processing {pdf_file.name}...")
        
        try:
            # Process document
            result = pdf_processor.process_document(str(pdf_file))
            
            if result['status'] == 'success':
                # Create chunks from the processed content
                chunks = text_chunker.process_document_chunks(result['content'])
                
                logger.info(f"Successfully processed {pdf_file.name}")
                logger.info(f"Generated {len(chunks)} chunks")
            else:
                logger.error(f"Failed to process {pdf_file.name}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")

if __name__ == "__main__":
    main()

