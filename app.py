import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import List, Dict, Any
import yaml
import pdfplumber

# Force CPU usage and set memory management
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import ollama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for better processing"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length <= self.chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.chunk_overlap//10:] if len(current_chunk) > self.chunk_overlap//10 else current_chunk
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class MilvusManager:
    """Handles Milvus database operations"""
    
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        # Force CPU usage to avoid CUDA memory issues
        self.embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
        self.dim = 1024  # BGE-M3 embedding dimension
        self.connection_alias = "default"
        self.setup_milvus()
    
    def setup_milvus(self):
        """Initialize Milvus connection and collection"""
        try:
            # Connect to Milvus
            connections.connect(
                alias=self.connection_alias,
                uri="./data/milvus_demo.db"  # Using Milvus Lite
            )
            
            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self.create_collection()
            
            self.collection = Collection(self.collection_name)
            logger.info(f"Connected to Milvus collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Milvus: {e}")
            raise
    
    def create_collection(self):
        """Create Milvus collection with proper schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_id", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with embeddings"
        )
        
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index
        index_params = {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection: {self.collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def store_document(self, filename: str, chunks: List[str]) -> bool:
        """Store document chunks in Milvus"""
        try:
            if not chunks:
                return False
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            if not embeddings:
                return False
            
            # Prepare data for insertion
            data = [
                chunks,  # text
                embeddings,  # embedding
                [filename] * len(chunks),  # filename
                list(range(len(chunks)))  # chunk_id
            ]
            
            # Insert data
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Stored {len(chunks)} chunks for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search parameters
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "filename", "chunk_id"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get("text"),
                        "filename": hit.entity.get("filename"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "score": hit.score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

class QueryProcessor:
    """Handles query processing and response generation"""
    
    def __init__(self, model_name: str = "mistral-nemo:latest"):
        self.model_name = model_name
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using Ollama"""
        try:
            # Prepare context
            context = "\n\n".join([chunk["text"] for chunk in context_chunks[:3]])  # Use top 3 chunks
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response using Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_ctx": 4096
                }
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="IntelliDoc - Document Q&A",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 IntelliDoc - Document Q&A System")
    st.markdown("Upload documents, store them in Milvus, and ask questions!")
    
    # Initialize components
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'milvus_manager' not in st.session_state:
        try:
            st.session_state.milvus_manager = MilvusManager()
        except Exception as e:
            st.error(f"Failed to initialize Milvus: {e}")
            st.stop()
    
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = QueryProcessor()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Extract text
                            text = st.session_state.processor.extract_text_from_pdf(tmp_file_path)
                            
                            if text:
                                # Chunk text
                                chunks = st.session_state.processor.chunk_text(text)
                                
                                if chunks:
                                    # Store in Milvus
                                    success = st.session_state.milvus_manager.store_document(
                                        uploaded_file.name, chunks
                                    )
                                    
                                    if success:
                                        st.success(f"✅ {uploaded_file.name} processed and stored!")
                                        st.info(f"Created {len(chunks)} chunks")
                                    else:
                                        st.error(f"❌ Failed to store {uploaded_file.name}")
                                else:
                                    st.error("❌ No text chunks created")
                            else:
                                st.error("❌ No text extracted from PDF")
                        
                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
    
    # Main query interface
    st.header("🔍 Ask Questions")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about your documents?"
    )
    
    if query:
        with st.spinner("Searching and generating response..."):
            # Search for relevant chunks
            search_results = st.session_state.milvus_manager.search_similar(query, top_k=5)
            
            if search_results:
                # Generate response
                response = st.session_state.query_processor.generate_response(query, search_results)
                
                # Display response
                st.markdown("### 💡 Answer:")
                st.write(response)
                
                # Display sources
                with st.expander("📑 Sources"):
                    for i, result in enumerate(search_results[:3], 1):
                        st.markdown(f"**Source {i}:** {result['filename']} (Chunk {result['chunk_id']})")
                        st.text(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
                        st.markdown("---")
            else:
                st.warning("No relevant information found. Please upload some documents first.")
    
    # Status section
    st.markdown("---")
    st.subheader("📊 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🔗 Milvus: Connected")
        
    with col2:
        try:
            # Check if Ollama model is available
            models = ollama.list()
            model_names = [model.model for model in models.models]
            if 'mistral-nemo:latest' in model_names or any('mistral-nemo' in name for name in model_names):
                st.info("🤖 Ollama: Ready")
            else:
                st.warning("⚠️ Ollama: mistral-nemo model not found")
                st.write("Available models:", model_names)
        except Exception as e:
            st.error("❌ Ollama: Not connected")
            st.write(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 