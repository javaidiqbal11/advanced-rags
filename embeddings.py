import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()
pdf_folder_path = "./data/pdfs/"
def load_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text
all_documents = []
for fn in os.listdir(pdf_folder_path):
    if fn.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, fn)
        pdf_text = load_pdf_to_text(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
        docs = text_splitter.split_text(pdf_text)
        documents = [Document(page_content=doc) for doc in docs]
        all_documents.extend(documents)
persist_dir = './vectorstore/'
embd = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma.from_documents(documents=all_documents, embedding=embd, persist_directory=persist_dir)

