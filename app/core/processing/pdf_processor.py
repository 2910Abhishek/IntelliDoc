import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']

    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a PDF and exists"""
        path = Path(file_path)
        return path.exists() and path.suffix.lower() in self.supported_formats

    def extract_metadata(self, file_path: str) -> Dict:
        """Extract metadata from PDF"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'number_of_pages': len(reader.pages),
                    'file_name': Path(file_path).name
                }
                return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}

    def extract_text(self, file_path: str) -> List[Dict]:
        """Extract text content from PDF with page numbers"""
        if not self.validate_file(file_path):
            raise ValueError("Invalid or unsupported file format")

        pages_content = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        pages_content.append({
                            'page_number': page_num,
                            'content': text.strip(),
                            'type': 'text'
                        })
            return pages_content
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def process_document(self, file_path: str) -> Dict:
        """Process PDF document and return both metadata and content"""
        try:
            metadata = self.extract_metadata(file_path)
            content = self.extract_text(file_path)
            
            return {
                'metadata': metadata,
                'content': content,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                'metadata': {},
                'content': [],
                'status': 'failed',
                'error': str(e)
            }