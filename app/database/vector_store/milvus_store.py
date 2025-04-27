import os
from pathlib import Path
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import yaml
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    """Custom LangChain Embeddings class for SentenceTransformer."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
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
                index_params={
                    "index_type": "FLAT",
                    "metric_type": "L2",
                    "params": {}
                },
                search_params={"metric_type": "L2"},
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
        """Add documents to the vector store."""
        try:
            self.vector_store.add_texts(texts)
            logger.info(f"Added {len(texts)} documents to collection")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(self, query: str, k: int = 2) -> List[Dict]:
        """Search for similar documents."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    @classmethod
    def from_config(cls, config_path: str) -> 'MilvusVectorStore':
        """Initialize MilvusVectorStore from a YAML config file."""
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
