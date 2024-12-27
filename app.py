import os
import re
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ensure proper memory allocation for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name='meta-llama/Meta-Llama-3-8B-Instruct'

# Initialize the tokenizer and model
def initialize_model(model_name= model_name , device='cuda'):
    print(100*"QA_llama")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    print(100*"QA_llama")
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

    Make sure:
    1. Each question addresses a different key aspect or detail from the text.
    2. The answers are well-explained, providing context or additional information when necessary.
    3. Both questions and answers should be clear and specific without direct references to phrases from the text.
    4. The questions should be open-ended (who, what, when, where, why, how) and encourage detailed responses.

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

# Extract and clean text from a PDF
def extract_and_clean_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with a space
        pages_text.append(text)
    
    # Debugging: Print the extracted text from PDF
    print("\n--- Extracted Text from PDF ---\n", pages_text)

    return pages_text

# Main function to process the PDF and generate QA pairs
def process_pdf(pdf_path, model_name= model_name, device='cuda'):
    tokenizer, model = initialize_model(model_name, device)
    pages_text = extract_and_clean_text_from_pdf(pdf_path)

    all_qa_pairs = []
    for page_text in pages_text:
        qa_pairs = process_text(tokenizer, model, page_text, device=device)
        all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs