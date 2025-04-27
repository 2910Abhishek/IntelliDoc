import pytest
from unittest.mock import patch, MagicMock
from app.core.llm.ollama_handler import OllamaHandler

@pytest.fixture
def mock_ollama():
    with patch('app.core.llm.ollama_handler.ollama') as mock:
        # Mock the list method with correct response structure
        mock.list.return_value = {
            'models': [
                {
                    'name': 'mistral-nemo',  # Updated model name
                    'modified_at': '2024-03-30T15:55:39.288102+05:30',
                    'size': 4563402752
                }
            ]
        }
        # Mock the generate method
        mock.generate.return_value = {
            'response': 'Test response'
        }
        yield mock

@pytest.fixture
def mock_llm(mock_ollama):
    with patch('app.core.llm.ollama_handler.OllamaHandler') as mock:
        instance = mock.return_value
        instance.model_name = "mistral-nemo"  # Updated model name
        instance.generate_response.return_value = {
            'status': 'success',
            'response': 'Test response',
            'model': 'mistral-nemo'  # Updated model name
        }
        yield mock

def test_initialization(mock_ollama):
    """Test OllamaHandler initialization"""
    handler = OllamaHandler()
    assert handler.model_name == "mistral"
    mock_ollama.list.assert_called_once()

def test_generate_response(mock_ollama):
    """Test response generation"""
    handler = OllamaHandler()
    response = handler.generate_response(
        prompt="What is Python?",
        context="Python is a programming language."
    )
    
    assert response['status'] == 'success'
    assert 'response' in response
    assert response['model'] == 'mistral'
    
    # Verify ollama.generate was called with correct parameters
    mock_ollama.generate.assert_called_once()
    call_args = mock_ollama.generate.call_args[1]
    assert call_args['model'] == 'mistral'
    assert 'Python is a programming language' in call_args['prompt']
    assert 'What is Python?' in call_args['prompt']

def test_generate_response_error(mock_ollama):
    """Test error handling in response generation"""
    mock_ollama.generate.side_effect = Exception("Test error")
    
    handler = OllamaHandler()
    response = handler.generate_response(prompt="Test prompt")
    
    assert response['status'] == 'error'
    assert 'error' in response
    assert 'Test error' in response['error']
