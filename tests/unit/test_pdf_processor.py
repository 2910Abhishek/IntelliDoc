import pytest
from pathlib import Path
from app.core.processing.pdf_processor import PDFProcessor
from app.utils.text_chunker import TextChunker

def test_pdf_processor_initialization():
    processor = PDFProcessor()
    assert processor.supported_formats == ['.pdf']

def test_text_chunker():
    chunk_size = 100
    chunk_overlap = 20
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Test with repetitive text
    test_text = "This is a test text. " * 10  # Each repeat is 19 chars
    chunks = chunker.split_into_chunks(test_text)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= chunk_size for chunk in chunks)
    
    # Test with empty text
    assert chunker.split_into_chunks("") == []
    
    # Test with text smaller than chunk size
    small_text = "Small text."
    small_chunks = chunker.split_into_chunks(small_text)
    assert len(small_chunks) == 1
    assert small_chunks[0] == small_text

    # Test with text exactly at chunk size
    exact_text = "x" * chunk_size
    exact_chunks = chunker.split_into_chunks(exact_text)
    assert len(exact_chunks) == 1
    assert len(exact_chunks[0]) == chunk_size

# Add more tests as needed
