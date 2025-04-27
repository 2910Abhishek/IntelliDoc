import pdfplumber
import PyPDF2
from pathlib import Path
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def process_pdf(self, uploaded_file):
        """Extract text and metadata from PDF, and chunk text."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Extract text with pdfplumber
            text_content = []
            with pdfplumber.open(tmp_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        text_content.append({
                            'page': page_num,
                            'text': text.strip()
                        })

            # Extract metadata with PyPDF2
            with open(tmp_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'pages': len(reader.pages),
                    'filename': uploaded_file.name
                }

            # Chunk text
            chunks = []
            for page in text_content:
                page_chunks = self.splitter.split_text(page['text'])
                for i, chunk in enumerate(page_chunks):
                    chunks.append({
                        'page': page['page'],
                        'chunk_id': i,
                        'text': chunk
                    })

            # Clean up
            Path(tmp_path).unlink()

            return {
                'status': 'success',
                'metadata': metadata,
                'chunks': chunks
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }