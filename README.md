# IntelliDoc - Document Q&A System

A professional document processing and question-answering system that uses Milvus vector database for storage and retrieval.

## Features

- 📄 **PDF Document Processing**: Upload and process PDF documents
- 🔍 **Text Chunking**: Intelligent text chunking with overlap for better context
- 🗄️ **Vector Storage**: Store document chunks in Milvus database with embeddings
- 🤖 **Query Interface**: Ask questions and get answers from your documents
- 🎯 **Semantic Search**: Find relevant information using semantic similarity
- 📊 **Real-time Status**: Monitor system status and document processing

## Prerequisites

- Python 3.8+
- Ollama
- Git

> **Note**: This application is configured to use CPU for processing to avoid GPU memory issues. This ensures stable operation on systems with limited VRAM.

## Quick Setup

### **Automated Setup (Recommended)**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd IntelliDoc

# 2. Run setup script (creates venv, installs dependencies)
./setup.sh

# 3. Run the application
./run.sh
```

### **Manual Setup**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd IntelliDoc

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# 3. Install PyTorch with CPU support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Install Ollama and download model
# Visit https://ollama.ai for Ollama installation
ollama pull mistral-nemo

# 6. Run the application
streamlit run app.py
```

## Usage

### **Running the Application**

**Option 1: Using run script (Recommended)**
```bash
# Make sure you're in the project directory
./run.sh
```

**Option 2: Manual run**
```bash
# Activate virtual environment first
source venv/bin/activate

# Then run the app
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### **Using the System**

1. **Upload Documents**
   - Use the sidebar to upload PDF files
   - Click "Process" for each document to extract text and create chunks
   - Documents are automatically stored in the Milvus database

2. **Ask Questions**
   - Enter your question in the main text input
   - The system will search for relevant chunks and generate an answer
   - View source chunks to see where the information came from

3. **Monitor Status**
   - Check the system status section to ensure Milvus and Ollama are connected

## Project Structure

```
IntelliDoc/
├── app.py              # Main application file
├── setup.sh            # Automated setup script
├── run.sh              # Application runner script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── LICENSE            # License file
├── .gitignore         # Git ignore file
├── venv/              # Virtual environment (created by setup)
└── data/              # Data directory (auto-created)
    └── milvus_demo.db # Milvus database (auto-created)
```

## Technical Details

- **PDF Processing**: Uses `pdfplumber` for text extraction
- **Text Chunking**: Splits text into 1000-character chunks with 200-character overlap
- **Embeddings**: Uses `BAAI/bge-m3` model for generating embeddings
- **Vector Database**: Milvus Lite for local vector storage
- **LLM**: Ollama with mistral-nemo model for response generation
- **UI**: Streamlit for the web interface
- **CPU Processing**: Optimized for CPU usage to ensure compatibility across all systems

## Configuration

The system uses sensible defaults:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Embedding model: BAAI/bge-m3
- LLM model: mistral-nemo
- Database: Local Milvus Lite instance

## Troubleshooting

### Common Issues

1. **Virtual Environment Issues**
   ```bash
   # If setup fails, try manual venv creation
   python3 -m venv venv
   source venv/bin/activate
   ./setup.sh
   ```

2. **PyTorch Installation Issues**
   ```bash
   # Reinstall PyTorch (CPU version)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Ollama not connected**
   - Make sure Ollama is running: `ollama serve`
   - Ensure mistral-nemo model is installed: `ollama pull mistral-nemo`

4. **Milvus connection issues**
   - The data directory will be created automatically
   - Database file is stored in `./data/milvus_demo.db`

5. **PDF processing errors**
   - Ensure PDF files are not corrupted
   - Check that files are readable PDFs (not scanned images)

## Dependencies

**Core Python packages:**
- `streamlit` - Web interface
- `pymilvus` - Milvus vector database client
- `sentence-transformers` - Text embeddings
- `pdfplumber` - PDF text extraction
- `ollama` - LLM integration
- `torch` - PyTorch (with CUDA support)

## License

MIT License - see LICENSE file for details.
