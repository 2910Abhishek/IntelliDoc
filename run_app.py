import os
import sys
import subprocess
import logging
import time
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all required directories and files exist."""
    required_dirs = [
        Path("data/vector_db"),
        Path("data/documents"),
    ]
    
    required_files = [
        Path("config/config.yaml"),
        Path("app/ui/streamlit_app.py")
    ]
    
    # Check directories
    for directory in required_dirs:
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    # Check files
    for file in required_files:
        if not file.exists():
            logger.error(f"Required file not found: {file}")
            return False
    
    return True

def check_ollama():
    """Check if Ollama is running."""
    try:
        import ollama
        client = ollama.Client()
        client.list()
        return True
    except Exception as e:
        logger.error("Ollama is not running. Please start Ollama first.")
        logger.error(f"Error: {str(e)}")
        return False

def run_streamlit():
    """Run the Streamlit application."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/ui/streamlit_app.py",
        "--server.fileWatcherType=none",
        "--server.headless=true",
        "--browser.serverAddress=localhost",
        "--server.address=localhost"
    ]
    
    try:
        # Start Streamlit process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for Streamlit to start
        time.sleep(2)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        logger.info("IntelliDoc is running at http://localhost:8501")
        logger.info("Press Ctrl+C to stop the application")
        
        # Keep the main process running
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Shutting down IntelliDoc...")
        process.terminate()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start IntelliDoc: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the application."""
    logger.info("Starting IntelliDoc...")
    
    # Check if running in virtual environment
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        logger.warning("Not running in a virtual environment. It's recommended to use one.")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please check the errors above.")
        sys.exit(1)
    
    # Check if Ollama is running
    if not check_ollama():
        logger.error("Please start Ollama first using 'ollama serve' command.")
        sys.exit(1)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()