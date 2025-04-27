# IntelliDoc - Multilingual Document Q&A System 📚

IntelliDoc is an intelligent document analysis system that enables users to upload PDF documents and ask questions about their content in multiple languages. The system uses advanced RAG (Retrieval-Augmented Generation) technology to provide accurate, context-aware responses.

## 🌟 Features

- 📄 PDF document processing and analysis
- 🔍 Intelligent question answering
- 🌐 Multilingual support
- 💾 Vector database storage for efficient retrieval
- 🖥️ User-friendly web interface
- 📊 Document chunk visualization
- 🚀 Real-time processing

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Language Models**: Mistral (via Ollama)
- **Embeddings**: BAAI/bge-m3
- **Vector Store**: Milvus
- **PDF Processing**: PyPDF2, PDFPlumber
- **ML Framework**: PyTorch, Sentence-Transformers

## 📋 Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally
- Git
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/IntelliDoc.git
   cd IntelliDoc
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   - Follow instructions at [Ollama's official website](https://ollama.ai/download)
   - Pull the required model:
     ```bash
     ollama pull mistral-nemo
     ```

5. **Create necessary directories**
   ```bash
   mkdir -p data/vector_db
   mkdir -p data/documents
   ```

## 🎯 Running the Application

1. **Start the application**
   ```bash
   python run_app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

1. **Upload Documents**
   - Use the sidebar to upload PDF documents
   - Multiple documents can be uploaded simultaneously
   - Progress indicators show processing status

2. **Ask Questions**
   - Enter your question in the main input field
   - Questions can be asked in any supported language
   - View source documents used for answers

3. **Document Management**
   - View processed documents in the sidebar
   - Access document metadata and statistics
   - Browse document chunks in the document viewer

## 🔧 Configuration

The system can be configured through `config/config.yaml`:

```yaml
vector_db:
  collection_name: "rag_collection"
  embedding_model: "BAAI/bge-m3"
  uri: "./data/vector_db/milvus_demo.db"

llm:
  model: "mistral-nemo"
  temperature: 0.7
  max_tokens: 500

processing:
  chunk_size: 1000
  chunk_overlap: 200
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Troubleshooting

**Common Issues:**

1. **Ollama Connection Error**
   ```bash
   # Ensure Ollama is running
   ollama serve
   ```

2. **Vector Store Issues**
   - Check if the data/vector_db directory exists and has write permissions
   - Clear the vector store:
     ```bash
     rm -rf data/vector_db/*
     ```

3. **Memory Issues**
   - Reduce chunk_size in config.yaml
   - Process fewer documents simultaneously

## 🔗 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Milvus Documentation](https://milvus.io/docs)

## 📧 Support

For support, please open an issue in the GitHub repository or contact the maintainers.
