import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from google.colab import userdata

# --------------------------------------------------
# API Key
# --------------------------------------------------

GOOGLEAPI = userdata.get("GOOGLE_APIKEY")

if not GOOGLEAPI:
    raise ValueError("GOOGLEAPI environment variable not found in Colab Secrets.")

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(title="LangChain RAG API")

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=GOOGLEAPI,
)

# --------------------------------------------------
# Sample Knowledge Base
# Replace with PDF/Text/Database later
# --------------------------------------------------

text = """
The Internet is a global system of interconnected computer networks that
uses TCP/IP to communicate.

The origins of the Internet date back to ARPANET, a project funded by
the United States Department of Defense.

ARPANET became operational in 1969 and laid the foundation for today's
modern Internet.
"""

documents = [Document(page_content=text)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLEAPI,
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}
"""
)

# --------------------------------------------------
# RAG Chain
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {"message": "LangChain RAG API is running"}

@app.post("/chat")
def chat(request: QueryRequest):
    answer = rag_chain.invoke(request.question)
    return {"answer": answer}
