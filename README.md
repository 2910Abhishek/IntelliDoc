# IntelliDoc Chatbot

A chatbot application that processes PDFs (scanned or digital), extracts text, stores it in a vector database, and answers questions based on the content.

## Project Setup

### Prerequisites
- Python 3.10+ (tested with 3.13 on Arch Linux)
- Virtual environment (`python -m venv venv && source venv/bin/activate`)
- Dependencies: See `requirements.txt` (to be created later)

### Directory Structure
IntelliDoc/
├── src/                # Source code
│   ├── ocr/           # OCR-related code
│   ├── vector_db/     # Vector database code
│   ├── qa/            # Q&A model code
│   └── utils/         # Helper functions
├── docs/              # Documentation
├── models/            # Pretrained models
├── data/              # Data storage
│   ├── raw/          # Uploaded PDFs
│   └── processed/    # Extracted text, embeddings
└── scripts/           # Deployment scripts
└── deploy/       # Docker and deployment files

text

Collapse

Wrap

Copy

### Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd IntelliDoc
Create __init__.py Files Since __init__.py files are excluded in .gitignore, recreate them:
bash

Collapse

Wrap

Copy
touch src/__init__.py src/ocr/__init__.py src/utils/__init__.py src/vector_db/__init__.py src/qa/__init__.py
These files mark directories as Python packages.
Install Dependencies
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Run the Application
bash

Collapse

Wrap

Copy
python -m src.app