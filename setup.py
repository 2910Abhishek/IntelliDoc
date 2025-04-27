from setuptools import setup, find_packages

setup(
    name="intellidoc",
    version="0.1.0",
    packages=find_packages(),
    package_data={'': ['*.yaml']},
    include_package_data=True,
    install_requires=[
        "PyPDF2",
        "pdfplumber",
        "streamlit",
        "ollama",
        "sentence-transformers",
        "langchain-core",
        "langchain-community",
        "langchain-milvus",
    ],
)
