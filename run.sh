#!/bin/bash

echo "🚀 Starting IntelliDoc..."
echo "================================"

# Force CPU usage to avoid CUDA memory issues
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the project."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
else
    echo "✅ Virtual environment is active"
fi

# Check if Ollama is running
if ! pgrep -f "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Check if mistral-nemo model is available
if ! ollama list | grep -q "mistral-nemo"; then
    echo "📦 Downloading mistral-nemo model..."
    ollama pull mistral-nemo
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "✅ Starting Streamlit application..."
echo "🌐 Application will be available at: http://localhost:8501"
echo "================================"

streamlit run app.py 