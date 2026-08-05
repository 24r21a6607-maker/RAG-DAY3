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


# --------------------------------------------------
# API Key from Render Environment Variable
# --------------------------------------------------

GOOGLE_APIKEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_APIKEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is missing"
    )


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="LangChain RAG API",
    version="1.0"
)


# --------------------------------------------------
# Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_APIKEY,
)


# --------------------------------------------------
# Knowledge Base
# --------------------------------------------------

text = """
The Internet is a global system of interconnected computer networks
that uses TCP/IP to communicate.

The origins of the Internet date back to ARPANET, a project funded by
the United States Department of Defense.

ARPANET became operational in 1969 and laid the foundation for today's
modern Internet.
"""


documents = [
    Document(
        page_content=text,
        metadata={"source": "internet_history"}
    )
]


# --------------------------------------------------
# Split Documents
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# --------------------------------------------------
# Gemini Embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_APIKEY,
)


# --------------------------------------------------
# FAISS Vector Database
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer only using the context.

If the answer is not available, say:
"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# RAG Chain
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


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
    return {
        "message": "LangChain RAG API is running"
    }


@app.post("/chat")
def chat(request: QueryRequest):

    answer = rag_chain.invoke(
        request.question
    )

    return {
        "answer": answer
    }
