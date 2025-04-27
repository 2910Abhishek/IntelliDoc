import unittest
from pathlib import Path
from app.database.vector_store.milvus_store import MilvusVectorStore

class TestMilvusVectorStore(unittest.TestCase):
    """Unit tests for MilvusVectorStore class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_path = Path("config/config.yaml")
        self.vector_store = MilvusVectorStore.from_config(self.config_path)
        self.sample_texts = [
            "This is a test document about AI.",
            "Another document discussing machine learning.",
            "A third document on natural language processing."
        ]
    
    def test_initialization(self):
        """Test MilvusVectorStore initialization."""
        self.assertIsNotNone(self.vector_store)
        self.assertEqual(self.vector_store.collection_name, "rag_collection")
        self.assertEqual(self.vector_store.dimension, 1024)
    
    def test_add_documents(self):
        """Test adding documents to Milvus."""
        try:
            self.vector_store.add_documents(self.sample_texts)
            self.assertTrue(True)  # If no exception, pass
        except Exception as e:
            self.fail(f"Adding documents failed: {e}")
    
    def test_search(self):
        """Test searching documents in Milvus."""
        self.vector_store.add_documents(self.sample_texts)
        query = "What is machine learning?"
        results = self.vector_store.search(query, k=2)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        for result in results:
            self.assertIn("text", result)
            self.assertIn("metadata", result)
            self.assertIsInstance(result["text"], str)
    
    def tearDown(self):
        """Clean up test environment."""
        # Optionally drop collection to reset state
        pass

if __name__ == "__main__":
    unittest.main()
