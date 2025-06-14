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
        # Force CPU usage and optimize for faster loading
        self.embedding_model = SentenceTransformer(
            'BAAI/bge-m3', 
            device='cpu',
            cache_folder='./data/models'  # Cache models locally
        )
        # Optimize model for inference
        self.embedding_model.eval()
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
    
    def search_similar(self, query: str, top_k: int = 5, filename_filter: str = None) -> tuple[List[Dict], Dict]:
        """Search for similar chunks with timing info and optional filename filtering"""
        timing_info = {}
        try:
            # Generate query embedding
            embed_start = time.time()
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            timing_info['embedding_time'] = time.time() - embed_start
            
            # Search parameters
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            # Add filename filter if specified
            filter_expr = None
            if filename_filter and filename_filter != "All Documents":
                filter_expr = f'filename == "{filename_filter}"'
                logger.info(f"Applying filter: {filter_expr}")
            
            # Ensure collection is loaded
            self.collection.load()
            
            # Perform search
            search_start = time.time()
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "filename", "chunk_id"],
                expr=filter_expr
            )
            timing_info['search_time'] = time.time() - search_start
            
            # Format results
            format_start = time.time()
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get("text"),
                        "filename": hit.entity.get("filename"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "score": hit.score
                    })
            timing_info['format_time'] = time.time() - format_start
            timing_info['filter_applied'] = filter_expr
            timing_info['results_count'] = len(formatted_results)
            
            return formatted_results, timing_info
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return [], {"error": str(e)}
    
    def get_available_documents(self) -> List[str]:
        """Get list of unique document filenames in the collection"""
        try:
            # Ensure collection is loaded
            self.collection.load()
            
            # Query to get all unique filenames
            results = self.collection.query(
                expr="",
                output_fields=["filename"],
                limit=1000  # Adjust based on expected number of documents
            )
            
            # Extract unique filenames
            filenames = list(set([result["filename"] for result in results]))
            logger.info(f"Found documents in database: {filenames}")
            return sorted(filenames)
            
        except Exception as e:
            logger.error(f"Error getting document list: {e}")
            return []
    
    def debug_collection_contents(self) -> Dict:
        """Debug function to show collection contents"""
        try:
            self.collection.load()
            
            # Get total count
            total_count = self.collection.num_entities
            
            # Get sample data
            sample_results = self.collection.query(
                expr="",
                output_fields=["filename", "chunk_id"],
                limit=10
            )
            
            return {
                "total_entities": total_count,
                "sample_data": sample_results
            }
            
        except Exception as e:
            logger.error(f"Error debugging collection: {e}")
            return {"error": str(e)}
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete all entities
            self.collection.delete(expr="")
            self.collection.flush()
            logger.info("Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
    
    def delete_document(self, filename: str) -> bool:
        """Delete a specific document from the collection"""
        try:
            # Delete entities with matching filename
            self.collection.delete(expr=f'filename == "{filename}"')
            self.collection.flush()
            logger.info(f"Deleted document: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return False

class QueryProcessor:
    """Handles query processing and response generation"""
    
    def __init__(self, model_name: str = "mistral-nemo:latest"):
        self.model_name = model_name
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> tuple[str, Dict]:
        """Generate response using Ollama with timing and multilingual support"""
        timing_info = {}
        try:
            # Prepare context
            context_start = time.time()
            context = "\n\n".join([chunk["text"] for chunk in context_chunks[:3]])  # Use top 3 chunks
            timing_info['context_prep_time'] = time.time() - context_start
            
            # Create multilingual prompt
            prompt = f"""Please answer the question based on the provided context. Respond in the same language as the question.

Context:
{context}

Question: {query}

Answer (respond in the same language as the question):"""
            
            # Generate response using Ollama
            gen_start = time.time()
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_ctx": 4096,
                    "num_predict": 512,  # Limit response length for faster generation
                    "top_k": 40,
                    "top_p": 0.9
                }
            )
            timing_info['generation_time'] = time.time() - gen_start
            
            return response['response'], timing_info
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", {"error": str(e)}

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
                            os.remove(tmp_file_path)
                                
        
        # Display processed documents
        st.markdown("---")
        st.header("📚 Processed Documents")
        
        sidebar_available_docs = st.session_state.milvus_manager.get_available_documents()
        if sidebar_available_docs:
            for i, doc in enumerate(sidebar_available_docs, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}.** {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc}", help=f"Delete {doc}"):
                        if st.session_state.milvus_manager.delete_document(doc):
                            st.success(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc}")
            
            # Management buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh List"):
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear All", type="secondary"):
                    if st.session_state.milvus_manager.clear_all_documents():
                        st.success("All documents cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear documents")
                        
            # Debug panel
            with st.expander("🔧 Debug Database"):
                if st.button("🔍 Show Database Contents"):
                    debug_info = st.session_state.milvus_manager.debug_collection_contents()
                    st.json(debug_info)
        else:
            st.info("No documents processed yet.")
    
    # Main query interface
    st.header("🔍 Ask Questions")
    
    # Document selector
    available_docs = st.session_state.milvus_manager.get_available_documents()
    if available_docs:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_document = st.selectbox(
                "📄 Select Document to Query:",
                ["All Documents"] + available_docs,
                help="Choose a specific document or search all documents"
            )
        with col2:
            doc_count = len(available_docs)
            st.metric("📚 Total Documents", doc_count)
            
        # Show document info
        if selected_document != "All Documents":
            st.info(f"🎯 Querying: **{selected_document}**")
        else:
            st.info(f"🔍 Searching across **all {doc_count} documents**")
    else:
        st.warning("📤 No documents found. Please upload some PDF files first.")
        selected_document = None
    
    # Language examples
    with st.expander("🌐 Multilingual Examples", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**English:**")
            st.code("What is the main topic of this document?")
            st.code("Can you summarize the key points?")
            
            st.markdown("**Spanish:**")
            st.code("¿Cuál es el tema principal de este documento?")
            st.code("¿Puedes resumir los puntos clave?")
            
            st.markdown("**French:**")
            st.code("Quel est le sujet principal de ce document?")
            st.code("Pouvez-vous résumer les points clés?")
        
        with col2:
            st.markdown("**German:**")
            st.code("Was ist das Hauptthema dieses Dokuments?")
            st.code("Können Sie die wichtigsten Punkte zusammenfassen?")
            
            st.markdown("**Hindi:**")
            st.code("इस दस्तावेज़ का मुख्य विषय क्या है?")
            st.code("क्या आप मुख्य बिंदुओं को सारांशित कर सकते हैं?")
            
            st.markdown("**Chinese:**")
            st.code("这个文档的主要主题是什么？")
            st.code("你能总结一下要点吗？")
    
    query = st.text_input(
        "Enter your question (any language):",
        placeholder="Ask in English, Spanish, French, German, Hindi, Chinese, or other languages..."
    )
    
    if query and available_docs:
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Search phase
            if selected_document != "All Documents":
                status_text.text(f"🔍 Searching in {selected_document}...")
            else:
                status_text.text("🔍 Searching for relevant information...")
            progress_bar.progress(20)
            
            search_results, search_timing = st.session_state.milvus_manager.search_similar(
                query, 
                top_k=5, 
                filename_filter=selected_document if selected_document != "All Documents" else None
            )
            progress_bar.progress(50)
            
            if search_results:
                # Generation phase
                status_text.text("🤖 Generating response...")
                progress_bar.progress(70)
                
                response, generation_timing = st.session_state.query_processor.generate_response(query, search_results)
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display response
                st.markdown("### 💡 Answer:")
                st.write(response)
                
                # Display performance metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🔤 Embedding", f"{search_timing.get('embedding_time', 0):.2f}s")
                with col2:
                    st.metric("🔍 Search", f"{search_timing.get('search_time', 0):.2f}s")
                with col3:
                    st.metric("🤖 Generation", f"{generation_timing.get('generation_time', 0):.2f}s")
                with col4:
                    st.metric("📄 Results", search_timing.get('results_count', 0))
                with col5:
                    total_time = (search_timing.get('embedding_time', 0) + 
                                 search_timing.get('search_time', 0) + 
                                 generation_timing.get('generation_time', 0))
                    st.metric("⏱️ Total", f"{total_time:.2f}s")
                
                # Debug information
                if search_timing.get('filter_applied'):
                    st.info(f"🎯 Filter Applied: {search_timing.get('filter_applied')}")
                else:
                    st.info("🔍 No filter applied - searching all documents")
                
                # Display sources
                with st.expander("📑 Sources"):
                    for i, result in enumerate(search_results[:3], 1):
                        st.markdown(f"**Source {i}:** {result['filename']} (Chunk {result['chunk_id']}) - Score: {result['score']:.3f}")
                        st.text(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
                        st.markdown("---")
            else:
                progress_bar.empty()
                status_text.empty()
                st.warning("No relevant information found. Please upload some documents first.")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error processing query: {str(e)}")
    
    # Status and Performance section
    st.markdown("---")
    st.subheader("📊 System Status & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🔗 Milvus: Connected")
        
        # Performance info
        with st.expander("⚡ Performance Info"):
            st.markdown("""
            **Expected Response Times:**
            - 🔤 Embedding: 0.1-0.5s
            - 🔍 Search: 0.01-0.1s  
            - 🤖 Generation: 2-10s
            - ⏱️ **Total: 2-11s**
            
            **Performance Tips:**
            - Keep app running (models stay loaded)
            - Use shorter, specific questions
            - First query slower (model loading)
            """)
        
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
        
        # Multilingual info
        with st.expander("🌐 Multilingual Support"):
            st.markdown("""
            **Supported Languages:**
            - English, Spanish, French, German
            - Hindi, Chinese, Japanese, Arabic
            - 100+ languages via BGE-M3
            
            **Tips:**
            - Ask in same language as documents
            - Cross-lingual search supported
            - Try example queries above
            """)

if __name__ == "__main__":
    main() 