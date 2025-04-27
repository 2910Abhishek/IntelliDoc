from typing import List, Dict
import re

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position for current chunk
            end = min(start + self.chunk_size, text_length)
            
            # If we're at the end of text, add final chunk and break
            if end >= text_length:
                chunk = text[start:end]
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append(chunk)
                break

            # Find the last period or newline within chunk_size limit
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)

            # If no good break point found, force break at chunk_size
            if break_point == -1 or break_point <= start:
                break_point = end - 1

            # Add chunk and move start position
            chunk = text[start:break_point + 1]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = break_point + 1 - self.chunk_overlap

        return chunks

    def process_document_chunks(self, pages: List[Dict]) -> List[Dict]:
        """Process document pages into chunks"""
        chunks = []

        for page in pages:
            page_chunks = self.split_into_chunks(page['content'])
            
            for chunk in page_chunks:
                chunks.append({
                    'content': chunk,
                    'page_number': page['page_number'],
                    'chunk_type': 'text'
                })

        return chunks
