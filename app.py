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



def extractTextFromPdf(pdf_path):

    # Open and read the PDF
    reader = PdfReader(pdf_path)

    pdf_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pdf_text += page_text + "\n"

    return pdf_text


def summarizePdfText(pdf_text):
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {
                "role": "user",
                "content": f"""
In 3-4 sentences of plain prose, tell me what this document is about and the overall approach it takes. Do not use bullet points and do not list the phases.

{pdf_text}
"""
            }
        ]
    )

    return response.output_text

if __name__=="__main__":
    pdf_path = r"C:\Users\mohit\Downloads\project-plan.pdf"

    pdf_text = extractTextFromPdf(pdf_path)

    summary = summarizePdfText(pdf_text)

    print(summary)
