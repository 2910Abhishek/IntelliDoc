import streamlit as st
from app.core.rag.pipeline import RAGPipeline
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_pipeline(config_path: str):
    """Initialize the RAGPipeline and cache it."""
    try:
        pipeline = RAGPipeline(str(config_path))
        logger.info("RAGPipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAGPipeline: {e}")
        st.error(f"Failed to initialize pipeline: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        config_path = Path("config/config.yaml")
        st.session_state.pipeline = initialize_pipeline(str(config_path))
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    return st.session_state.pipeline is not None

def main():
    st.set_page_config(
        page_title="IntellDoc - Multilingual Document Q&A",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 IntellDoc - Multilingual Document Q&A")
    st.markdown("---")

    if not initialize_session_state():
        st.stop()

    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )

        if uploaded_files and st.session_state.pipeline:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [doc['metadata']['filename'] for doc in st.session_state.documents]:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = st.session_state.pipeline.process_document(uploaded_file)
                        if result['status'] == 'success':
                            st.session_state.documents.append(result)
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}: {result['error']}")

        # Display processed documents
        if st.session_state.documents:
            st.markdown("### 📚 Processed Documents")
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc['metadata']['filename']}"):
                    st.json(doc['metadata'])

    # Main content area
    if not st.session_state.documents:
        st.info("👆 Please upload PDF documents using the sidebar to get started.")
        return

    # Question answering section
    st.header("🤔 Ask Questions About Your Documents")
    question = st.text_input("Enter your question:", help="Ask a question in any supported language")

    if question and st.session_state.pipeline:
        with st.spinner("🔍 Searching through documents..."):
            response = st.session_state.pipeline.query(question)
            if response['status'] == 'success':
                st.markdown("### 💡 Answer:")
                st.write(response['response'])
                with st.expander("📑 Source Chunks"):
                    for source in response['sources']:
                        st.text(source['text'])
            else:
                st.error(f"Failed to generate response: {response['error']}")

    # Document viewer section
    st.markdown("---")
    st.header("📑 Document Viewer")
    selected_doc = st.selectbox(
        "Select a document to view:",
        options=[doc['metadata']['filename'] for doc in st.session_state.documents]
    )

    if selected_doc and st.session_state.pipeline:
        doc = next(doc for doc in st.session_state.documents if doc['metadata']['filename'] == selected_doc)
        for chunk in doc['chunks']:
            with st.expander(f"Page {chunk['page']} - Chunk {chunk['chunk_id']}"):
                st.text(chunk['text'])

if __name__ == "__main__":
    main()