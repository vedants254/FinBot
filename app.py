# Fix for PyTorch/Streamlit compatibility issue
try:
    import torch
    # Monkey patch to avoid the __path__ issue
    if hasattr(torch, '_classes'):
        torch._classes.__path__ = []
except (ImportError, AttributeError):
    pass

from main import load_documents, chunk_documents, conv_retrieval_chain
import streamlit as st
from langchain.schema import Document 
from html_templates import css, bot_template, user_template
from dotenv import load_dotenv
load_dotenv()
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
