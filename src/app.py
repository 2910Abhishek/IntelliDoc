import gradio as gr
import os
import shutil
from src.ocr.extract_text import extract_text_from_pdf
from src.utils.preprocess import preprocess_text
from src.vector_db.store_embeddings import store_in_chroma
from src.qa.query_chroma import query_chroma

def process_pdf_and_query(pdf_file, question):
    """
    Process a PDF and answer a question based on its content.
    Args:
        pdf_file: Path to the uploaded PDF file from Gradio.
        question (str): User’s question.
    Returns:
        str: Answer to the question.
    """
    # Define paths
    raw_dir = "data/raw/"
    processed_dir = "data/processed/"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Save uploaded PDF (pdf_file is a temp file path)
    pdf_path = os.path.join(raw_dir, os.path.basename(pdf_file))
    shutil.copy(pdf_file, pdf_path)  # Copy temp file to raw_dir
    
    # Extract text
    text_file = extract_text_from_pdf(pdf_path, processed_dir)
    
    # Preprocess and store in Chroma
    store_in_chroma(text_file, processed_dir, collection_name="pdf_chunks")
    
    # Answer the question
    answer = query_chroma(question)
    return answer

# Gradio interface
with gr.Blocks(title="IntelliDoc Chatbot") as demo:
    gr.Markdown("# IntelliDoc Chatbot")
    gr.Markdown("Upload a PDF and ask questions about its content.")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        question_input = gr.Textbox(label="Ask a Question", placeholder="What’s in the PDF?")
    
    submit_btn = gr.Button("Submit")
    output = gr.Textbox(label="Answer")
    
    submit_btn.click(
        fn=process_pdf_and_query,
        inputs=[pdf_input, question_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()