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
from langchain_community.vectorstores import Chroma
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

# In your app.py, after load_dotenv()
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
            # Extract text
            page_text = page.extract_text() or ""
            text_all += page_text + "\n"
            
            # Extract tables as text
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
    vectorstore = Chroma.from_documents(doc, embeddings)
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True,output_key='answer')

    llm=Ollama(model='llama2')
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
#hanhle user input
def handle_input(user_question):
    if st.session_state.conversation is None:
        st.error('Please upload the files first')
        return
    response=st.session_state.conversation({'question': user_question})
    st.session_state.chat_history=response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 ==0:
            st.write(user_template.replace(
                '{{MSG}}',message.content), unsafe_allow_html=True
            )
        else:
            st.write(bot_template.replace(
                '{{MSG}}', message.content), unsafe_allow_html=True
            )

import time 
#Final implementation 
def main():
    st.set_page_config(page_title='FinBoT: Financial Analysis made easy!')
    st.write(css,unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation=None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=None
    st.header('Financial Analysis made easy!')
    user_question=st.text_input('Ask a Question about your documents')
    if user_question:
        handle_input(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs=st.file_uploader(
            "Upload your reports(PDFs) here and click on 'Process'", accept_multiple_files=True
        )

        if st.button('Process'):
            start=time.process_time()
            with st.spinner('Processing'):
                docs = load_documents(pdf_docs)
                chunks = chunk_documents(docs)
                chain, vectorstore = conv_retrieval_chain(chunks)
                print('Response time:',time.process_time()-start)
                st.session_state.conversation=chain
                st.success("Documents processed successfully!")
        


if __name__=='__main__':
    main()
