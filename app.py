import os

from fastapi import FastAPI
from langserve import add_routes

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

# -------------------------------------------------------
# API Key
# -------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_APIKEY")

if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_APIKEY environment variable")

# -------------------------------------------------------
# LLM
# -------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_APIKEY")
)
# -------------------------------------------------------
# Documents
# -------------------------------------------------------

text = """
The Internet is a global system of interconnected computer networks
that uses the Internet protocol suite (TCP/IP).

ARPANET was the first precursor of the modern Internet.

The National Science Foundation Network (NSFNET)
expanded Internet access during the 1980s.

Commercial Internet Service Providers made the
Internet available to the public in the early 1990s.
"""

docs = [Document(page_content=text)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

splits = splitter.split_documents(docs)

# -------------------------------------------------------
# Embeddings
# -------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_APIKEY,
)

vectorstore = FAISS.from_documents(splits, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------------
# Prompt
# -------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
Answer the question only using the provided context.

Context:
{context}

Question:
{question}
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------------
# FastAPI
# -------------------------------------------------------

app = FastAPI(
    title="LangServe RAG API",
    version="1.0",
    description="RAG API using LangServe + Gemini",
)

add_routes(
    app,
    chain,
    path="/rag",
)

@app.get("/")
def home():
    return {
        "message": "LangServe is running!",
        "playground": "/rag/playground",
        "invoke": "/rag/invoke",
        "stream": "/rag/stream",
    }
