import gradio as gr
import os
import shutil
from src.ocr.extract_text import extract_text_from_pdf
from src.vector_db.store_embeddings import store_in_chroma
from src.qa.query_chroma import query_chroma

def process_pdf_and_query(pdf_file, question):
    """
    Process a PDF and answer a question, reusing extracted text if it exists.
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
    
    # Get PDF filename and paths
    pdf_name = os.path.basename(pdf_file)
    pdf_path = os.path.join(raw_dir, pdf_name)
    text_file = os.path.join(processed_dir, f"{os.path.splitext(pdf_name)[0]}_extracted.txt")
    
    # Copy uploaded PDF if it’s new
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) != os.path.getsize(pdf_file):
        shutil.copy(pdf_file, pdf_path)
    
    # Extract text only if it doesn’t exist
    if not os.path.exists(text_file):
        text_file = extract_text_from_pdf(pdf_path, processed_dir)
        # Store in Chroma only when extracting new text
        store_in_chroma(text_file, processed_dir, collection_name="pdf_chunks")
    else:
        print(f"Reusing existing extracted text: {text_file}")
    
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