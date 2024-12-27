import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

parser = LlamaParse(
    api_key= API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# sync
documents = parser.load_data("BS_2_table.pdf")

file_name = "BS_2.md"
with open(file_name, 'w') as file:
    file.write(documents[0].text)