from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader

# Load API key from .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Send message to LLM
def sayHiToLLM():
    response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "user", "content": "Hi"}
    ]
    )
    # Print LLM reply
    print(response.choices[0].message.content)


# Use r before a Windows path to avoid backslash errors
pdf_path = r"C:\Users\mohit\Downloads\project-plan.pdf"

# Open and read the PDF
reader = PdfReader(pdf_path)

pdf_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        pdf_text += page_text + "\n"
# Print extracted PDF text
print(pdf_text)


