import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2  # For PDF extraction
import fitz  # For OCR (using PyMuPDF)

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("AIzaSyDc4XGLLxn8Vw-Ia4Q_zwW1IRaKK6sR1j8")
if api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the API with the key
genai.configure(api_key=api_key)

def evaluate_pdf(pdf_file_path):
    """Evaluates the text extracted from a PDF using Gemini API and returns a score."""

    # Extract text from PDF using PyPDF2 for simple PDFs
    text = extract_text_from_pdf(pdf_file_path)
    
    # Create the model with updated generation configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Create the prompt
    prompt = f"Use NLP to evaluate the following text and assign a score out of 10 based on correctness and quality.\n{text}"

    # Send the prompt to the model and get the response
    response = model.generate_content(prompt)
    
    if response is None:
        raise ValueError("No response received from the model")

    score = parse_score_from_response(response)
    return score

def extract_text_from_pdf(pdf_file_path):
    """Extracts text from a PDF file, using OCR if necessary."""
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
                
            # If PyPDF2 fails to extract text (e.g., scanned PDF), use OCR
            if not text.strip():
                print("PyPDF2 failed to extract text. Trying OCR...")
                text = extract_text_with_ocr(pdf_file_path)
                
    except PyPDF2.utils.PdfReadError:
        # Handle cases where PyPDF2 fails
        print("PyPDF2 encountered an error. Trying OCR...")
        text = extract_text_with_ocr(pdf_file_path)

    return text

def extract_text_with_ocr(pdf_file_path):
    """Uses OCR (PyMuPDF) to extract text from a scanned PDF."""
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Use "text" to extract all text
    return text

def parse_score_from_response(response):
    """Parses the score from the response text."""
    # Assuming the response contains the score in a text format
    score_text = response.text.strip()
    try:
        score = float(score_text)
        return score
    except ValueError:
        print(f"Error parsing score: '{score_text}' is not a valid number.")
        return None 

# Example usage
if __name__ == "__main__":
    # Get PDF file path from user
    pdf_file_path = input("Enter the path to the PDF file: ")

    # Evaluate the PDF and print the score
    score = evaluate_pdf(pdf_file_path)
    if score is not None:
        print(f"The score for the PDF is: {score}")
