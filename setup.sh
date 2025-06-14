#!/bin/bash

echo "🚀 IntelliDoc Setup Script"
echo "=========================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CPU support (to avoid CUDA memory issues)
echo "💻 Installing PyTorch with CPU support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed."
    echo "📖 Please install Ollama from: https://ollama.ai"
    echo "   Then run: ollama pull mistral-nemo"
else
    echo "✅ Ollama is installed"
    
    # Check if mistral-nemo model is available
    if ! ollama list | grep -q "mistral-nemo"; then
        echo "📦 Downloading mistral-nemo model..."
        ollama pull mistral-nemo
    else
        echo "✅ mistral-nemo model is available"
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo "=========================="
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the app: ./run.sh"
echo "   OR: streamlit run app.py"
echo ""
echo "Application will be available at: http://localhost:8501" 