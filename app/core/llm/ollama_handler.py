import logging
from typing import Dict, Optional
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaHandler:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify if the model is available locally, if not pull it."""
        try:
            logger.info(f"Checking for model {self.model_name}...")
            # List models to check if our model exists
            models = ollama.list()
            model_exists = any(model['name'] == self.model_name for model in models['models'])
            
            if not model_exists:
                logger.info(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
                logger.info(f"Successfully pulled model {self.model_name}")
        except Exception as e:
            logger.error(f"Error verifying/pulling model: {e}")
            raise

    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None, 
                         temperature: float = 0.7) -> Dict:
        """
        Generate a response using the Ollama model.
        
        Args:
            prompt: The main question or prompt
            context: Additional context from retrieved documents
            temperature: Controls randomness in the response (0.0 to 1.0)
        """
        try:
            system_prompt = """You are a helpful AI assistant. 
            If context is provided, use it to answer the question accurately.
            If no context is provided or the context is irrelevant, 
            say that you don't have enough information to answer accurately."""

            full_prompt = f"""Context: {context if context else 'No context provided'}

Question: {prompt}

Please provide a clear and concise answer based on the context provided."""

            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                system=system_prompt,
                temperature=temperature
            )

            return {
                'status': 'success',
                'response': response['response'],
                'model': self.model_name
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'model': self.model_name
            }