import sys
import os

# Fix for PyTorch/Streamlit compatibility issue
try:
    import torch
    # Monkey patch to avoid the __path__ issue
    if hasattr(torch, '_classes'):
        torch._classes.__path__ = []
except (ImportError, AttributeError):
    pass


from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.llms import Ollama , HuggingFacePipeline 
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pdfplumber
from langchain.schema import Document 
from html_templates import css, bot_template, user_template
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'gigachad'  
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')





#load and parse PDFs (extract text & tables as text)

def extract_text_from_pdf(pdf_file):
    text_all = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            #etract text
            page_text = page.extract_text() or ""
            text_all += page_text + "\n"
            
            #extract tables as text
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join([str(cell) if cell else "" for cell in row])
                    text_all += row_text + "\n"
    return text_all

def load_documents(uploaded_files):
    docs = []
    for pdf in uploaded_files:
        if pdf.name.lower().endswith(".pdf"):
            content = extract_text_from_pdf(pdf)
            docs.append(Document(page_content=content, metadata={"source": pdf.name}))
        elif pdf.name.lower().endswith(".txt"):
            with open(pdf, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": pdf.name}))
        else:
            raise ValueError(f"Unsupported file format: {pdf.name}")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(splits):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i}
                )
            )
    return chunks


def conv_retrieval_chain(doc):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    #vectorstore = Chroma.from_documents(doc, embeddings)
    vectorstore = FAISS.from_documents(doc, embeddings)
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True,output_key='answer')

    llm=Ollama(model='llama3.1:8b')
    prompt=ChatPromptTemplate.from_template(
        '''
        You are an intelligent Financial analyst.
        Use the following context from company reports to answer the question below. 
        Be detailed, accurate, and use relevant financial reasoning. 
        <context>
        {context}
        </context>
        question={question}
        '''
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}

    )
    return chain, vectorstore
