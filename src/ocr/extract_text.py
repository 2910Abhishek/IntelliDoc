import os
from paddleocr import PaddleOCR
import pdfplumber
from pathlib import Path

def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extract text from a PDF (scanned or digital) and save it to a text file.
    Args:
        pdf_path (str): Path to the input PDF.
        output_dir (str): Directory to save extracted text.
    Returns:
        str: Path to the output text file.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Output file name
    pdf_name = Path(pdf_path).stem
    output_file = os.path.join(output_dir, f"{pdf_name}_extracted.txt")
    
    # Initialize PaddleOCR (CPU-only)
    ocr = PaddleOCR(use_gpu=False, lang='en')  # English language
    
    # Try digital extraction first (faster if text is embedded)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            if text.strip():  # If text is found, use it
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted digital text from {pdf_path} to {output_file}")
                return output_file
    except Exception as e:
        print(f"Digital extraction failed: {e}. Falling back to OCR.")
    
    # Fallback to OCR for scanned PDFs
    result = ocr.ocr(pdf_path, cls=True)  # cls=True for better orientation detection
    text = ''
    for page in result:
        for line in page:
            text += line[1][0] + '\n'  # line[1][0] is the text content
    
    # Save extracted text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Extracted OCR text from {pdf_path} to {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage
    input_pdf = "data/raw/Abhishek.pdf"  # Replace with your test PDF
    output_dir = "data/processed/"
    extract_text_from_pdf(input_pdf, output_dir)