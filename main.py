import os
import re
#import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from llama_parse import LlamaParse
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

# Initialize llama parser
parser = LlamaParse(
    api_key=API_KEY,  # API key loaded from .env
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# Ensure proper memory allocation for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def login_to_huggingface():
    token = os.getenv('HUGGINGFACE_API_TOKEN')  # Fetch the token from the environment variables
    if not token:
        raise ValueError("Hugging Face API token is not set in environment variables.")
    login(token)
    print("Logged in to Hugging Face")

login_to_huggingface()

# Model name for the transformer
model_name = 'meta-llama/Llama-3.2-8B-Instruct'



# Initialize the tokenizer and model
def initialize_model(model_name=model_name, device='cuda'):
    print(100 * "QA_llama")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    print(100 * "QA_llama")
    return tokenizer, model

# Chunk the input text into smaller parts
def chunk_text(tokenizer, text, chunk_size=150):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Generate question-answer pairs from a text chunk using the custom prompt
def generate_qa_from_chunk(tokenizer, model, chunk, device='cuda'):
    prompt = f"""
    You are tasked with generating high-quality, meaningful question-answer pairs from the following text. The questions should cover all key points in the text, be diverse, and not repetitive. The answers should be detailed, conversational, accurate, and provide explanations where necessary. Avoid using wording that directly refers back to the text (like "according to the text"). Ensure each question is unique and comprehensive.

    Here is the text to generate questions and answers from:

    {chunk}

    Format your response like this:
    Question: [Insert question here]
    Answer: [Insert detailed, conversational answer here]

    Start generating the question-answer pairs now.
    """

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debugging: Print the generated text from the model
    print("\n--- Generated Text ---\n", generated_text)

    # Initialize variables for storing Q&A pairs
    qa_pairs = []
    question, answer = None, None

    # Split the generated text by lines and extract Q&A pairs using robust matching
    for line in generated_text.split("\n"):
        if re.match(r'^(Question:|Q:|q:|que:|QUE:|Que:)', line):
            if question and answer:
                qa_pairs.append((question, answer))
            question = re.sub(r'^(Question:|Q:|q:|que:|QUE:|Que:)', '', line).strip()
            answer = None
        elif re.match(r'^(Answer:|A:|a:|ans:|ANS:|Ans:)', line):
            answer = re.sub(r'^(Answer:|A:|a:|ans:|ANS:|Ans:)', '', line).strip()
        elif answer is not None:
            answer += " " + line.strip()

    if question and answer:
        qa_pairs.append((question, answer))

    # Debugging: Print the extracted Q&A pairs
    print("\n--- Extracted Q&A Pairs ---\n", qa_pairs)

    return qa_pairs

# Process the entire text to generate QA pairs
def process_text(tokenizer, model, text, chunk_size=150, device='cuda'):
    chunks = chunk_text(tokenizer, text, chunk_size)
    
    # Debugging: Print the chunks being processed
    print("\n--- Text Chunks ---\n", chunks)
    
    qa_list = []
    for chunk in chunks:
        qa_pairs = generate_qa_from_chunk(tokenizer, model, chunk, device)
        qa_list.extend(qa_pairs)  # Flatten the nested list
    return qa_list

# Read and clean text from a Markdown (.md) file
def read_and_clean_text_from_md(md_path):
    with open(md_path, 'r') as file:
        text = file.read()
    
    # Clean the text by removing excess whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with a space
    
    # Debugging: Print the cleaned text from MD file
    print("\n--- Cleaned Text from MD File ---\n", text)
    
    return text

# Parse the PDF into a Markdown file using LlamaParser
def parse_pdf_with_llama(pdf_path, output_md="output.md"):
    documents = parser.load_data(pdf_path)
    with open(output_md, 'w') as file:
        file.write(documents[0].text)

def save_qa_pairs_to_file(qa_pairs, output_file="qa_pairs.txt"):
    with open(output_file, 'w') as file:
        for idx, (question, answer) in enumerate(qa_pairs):
            file.write(f"Q{idx + 1}: {question}\n")
            file.write(f"A{idx + 1}: {answer}\n\n")
    print(f"QA pairs saved to {output_file}")

# Main function to parse PDF and generate QA pairs
def process_parsed_pdf(pdf_path, model_name=model_name, device='cuda'):
    # Step 1: Parse the PDF to a Markdown (.md) file
    md_file = "output.md"
    parse_pdf_with_llama(pdf_path, output_md=md_file)

    # Step 2: Read and clean the text from the .md file
    cleaned_text = read_and_clean_text_from_md(md_file)

    # Step 3: Initialize the model
    tokenizer, model = initialize_model(model_name, device)

    # Step 4: Process the text to generate QA pairs
    qa_pairs = process_text(tokenizer, model, cleaned_text, device=device)
    
    save_qa_pairs_to_file(qa_pairs, "qa_pairs.txt")

    return qa_pairs

# Example usage
pdf_path = "BS_2_table.pdf"

qa_pairs = process_parsed_pdf(pdf_path)

# Print the final QA pairs
print("\n--- Final QA Pairs ---\n", qa_pairs)
