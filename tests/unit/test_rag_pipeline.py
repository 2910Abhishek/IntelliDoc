import pytest
from unittest.mock import Mock, patch
from app.core.rag.rag_pipeline import RAGPipeline

@pytest.fixture
def mock_vector_store():
    with patch('app.database.vector_store.milvus_store.MilvusVectorStore') as mock:
        mock.from_config.return_value = Mock()
        mock.from_config.return_value.search.return_value = [
            {'text': 'Sample context 1', 'metadata': {}},
            {'text': 'Sample context 2', 'metadata': {}}
        ]
        yield mock

@pytest.fixture
def mock_llm():
    with patch('app.core.llm.ollama_handler.OllamaHandler') as mock:
        mock.return_value.generate_response.return_value = {
            'status': 'success',
            'response': 'Test response',
            'model': 'mistral'
        }
        yield mock

def test_rag_pipeline_initialization(mock_vector_store, mock_llm):
    """Test RAG pipeline initialization"""
    pipeline = RAGPipeline()
    assert pipeline.vector_store is not None
    assert pipeline.llm is not None
    assert pipeline.text_chunker is not None

def test_add_documents(mock_vector_store, mock_llm):
    """Test adding documents to RAG pipeline"""
    pipeline = RAGPipeline()
    
    test_docs = [{
        'content': [
            {
                'page_number': 1,
                'content': 'Test content',
                'type': 'text'
            }
        ],
        'metadata': {'title': 'Test Doc'}
    }]
    
    result = pipeline.add_documents(test_docs)
    assert result['status'] == 'success'

def test_query(mock_vector_store, mock_llm):
    """Test querying the RAG pipeline"""
    pipeline = RAGPipeline()
    
    result = pipeline.query("What is Python?")
    
    assert result['status'] == 'success'
    assert 'answer' in result
    assert 'context' in result
    assert 'model' in result