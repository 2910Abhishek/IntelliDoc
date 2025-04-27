import os
import subprocess
from pathlib import Path
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories for vector database storage."""
    directories = [
        Path("data/vector_db"),
        Path("config"),
        Path("logs")
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/Verified directory: {directory}")

def install_dependencies(requirements_path: str = "requirements.txt"):
    """Install dependencies from requirements.txt."""
    try:
        subprocess.run(["pip", "install", "-r", requirements_path], check=True)
        logger.info(f"Installed dependencies from {requirements_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise

def create_default_config(config_path: str = "config/config.yaml"):
    """Create default Milvus configuration if it doesn't exist."""
    default_config = {
        "vector_db": {
            "collection_name": "rag_collection",
            "embedding_model": "BAAI/bge-m3",
            "dimension": 1024,
            "uri": "./data/vector_db/milvus_demo.db"
        }
    }
    config_path = Path(config_path)
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(default_config, f)
        logger.info(f"Created default config at {config_path}")
    else:
        logger.info(f"Config already exists at {config_path}")

def download_embedding_model():
    """Download the embedding model if not already present."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Downloading embedding model BAAI/bge-m3...")
        SentenceTransformer("BAAI/bge-m3")
        logger.info("Embedding model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download embedding model: {e}")
        raise

def main():
    """Run Milvus setup tasks."""
    try:
        # Ensure project directories
        ensure_directories()
        
        # Install dependencies
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            install_dependencies(str(requirements_path))
        else:
            logger.warning(f"{requirements_path} not found. Skipping dependency installation.")
        
        # Download embedding model
        download_embedding_model()
        
        # Create default configuration
        create_default_config()
        
        logger.info("Milvus setup completed successfully.")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()
