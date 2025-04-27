import os
from pathlib import Path
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import yaml
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    """Custom LangChain Embeddings class for SentenceTransformer."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
        return embedding

class MilvusVectorStore:
    """Manages Milvus vector database for storing and searching document embeddings."""
    
    def __init__(self, 
                 collection_name: str = "rag_collection", 
                 embedding_model: str = "BAAI/bge-m3",
                 uri: str = "./data/vector_db/milvus_demo.db"):
        """
        Initialize Milvus vector store with LangChain integration.
        
        Args:
            collection_name (str): Name of the Milvus collection.
            embedding_model (str): Hugging Face model for embeddings.
            uri (str): Milvus connection URI (local SQLite or Zilliz Cloud endpoint).
        """
        self.collection_name = collection_name
        self.uri = uri
        
        # Load embedding model
        try:
            self.embedder = SentenceTransformerEmbeddings(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize LangChain Milvus vector store
        try:
            self.vector_store = Milvus(
                embedding_function=self.embedder,
                collection_name=self.collection_name,
                connection_args={"uri": self.uri},
                index_params={"index_type": "FLAT", "metric_type": "L2"},
                primary_field="pk",
                text_field="text",
                vector_field="vector",
                auto_id=True
            )
            logger.info(f"Initialized Milvus vector store with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            raise
    
    def add_documents(self, texts: List[str]) -> None:
        """
        Embed and store documents in Milvus.
        
        Args:
            texts (List[str]): List of text chunks to embed and store.
        """
        try:
            # Add to vector store (embeddings are handled by langchain_milvus)
            self.vector_store.add_texts(texts)
            logger.info(f"Added {len(texts)} documents to Milvus collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """
        Search for relevant documents based on query.
        
        Args:
            query (str): User query to embed and search.
            k (int): Number of top results to return.
        
        Returns:
            List[dict]: List of retrieved documents with text and metadata.
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} documents for query")
            return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    @classmethod
    def from_config(cls, config_path: str) -> 'MilvusVectorStore':
        """
        Initialize MilvusVectorStore from a YAML config file.
        
        Args:
            config_path (str): Path to YAML configuration file.
        
        Returns:
            MilvusVectorStore: Configured vector store instance.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            vector_db_config = config.get('vector_db', {})
            return cls(
                collection_name=vector_db_config.get('collection_name', 'rag_collection'),
                embedding_model=vector_db_config.get('embedding_model', 'BAAI/bge-m3'),
                uri=vector_db_config.get('uri', './data/vector_db/milvus_demo.db')
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

# Example usage (for testing)
if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    # Initialize vector store
    vector_store = MilvusVectorStore.from_config(config_path)
    
    # Sample documents
    sample_texts = [
        "This is a sample document about AI.",
        "Another document discussing machine learning.",
        "A third document on natural language processing."
    ]
    
    # Add documents
    vector_store.add_documents(sample_texts)
    
    # Search
    query = "What is machine learning?"
    results = vector_store.search(query, k=2)
    for result in results:
        print(f"Text: {result['text']}, Metadata: {result['metadata']}")