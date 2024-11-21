import numpy as np
import streamlit as st
import re
import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

# STATE: Weblist reference for RAG
weblist = []
if 'weblist' not in st.session_state:
    st.session_state['weblist'] = weblist

# RAG Logic
# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer
    
def rag_response(user_prompt):
    docs = [WebBaseLoader(url).load() for url in st.session_state['weblist']]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs_list)

    vector_store = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    retreiver = vector_store.as_retriever(k=4)

    prompt = PromptTemplate(
    template="""You are CHAT WITH WEBSITE an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    rag_chain = prompt | llm | StrOutputParser()

    rag_application = RAGApplication(retriever=retreiver, rag_chain=rag_chain)

    question = user_prompt
    answer = rag_application.run(question=question)

    return answer


# STATE: Chat Messages list
messages = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = messages

def is_valid_url(str):
    # Regex to check valid URL 
    regex = ("((http|https)://)(www.)?" +
             "[a-zA-Z0-9@:%._\\+~#?&//=]" +
             "{2,256}\\.[a-z]" +
             "{2,6}\\b([-a-zA-Z0-9@:%" +
             "._\\+~#?&//=]*)")
    
    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty 
    # return false
    if (str == None):
        return False

    # Return if the string 
    # matched the ReGex
    if(re.search(p, str)):
        return True
    else:
        return False

with st.sidebar:
    st.title("Chat with Websites 🌐")
    st.divider()
    with st.form(key="add_web",clear_on_submit=True,enter_to_submit=True):
        input_val = st.text_input(label="Tambahkan URL Website untuk sumber informasi")
        submitted = st.form_submit_button(label="Tambah")

        if submitted:
            is_valid_input = is_valid_url(input_val)
            if is_valid_input:
                st.session_state['weblist'].append(input_val)
            else:
                st.error("Alamat url yang anda masukan tidak valid, silahkan masukan ulang url website yang valid untuk sumber informasi.")


    st.title("Daftar Alamat Website:")
    if st.session_state['weblist']:
        for web_url in st.session_state['weblist']:
            st.markdown(f"- {web_url}")
    else:
        st.info("Anda belum menambahkan website untuk sumber informasi")

if len(st.session_state['weblist']) == 0:
    st.header("<< Please input valid web reference url")    
    st.markdown(
        '''
        # Aplikasi Web RAG "Chat with Website
            1. Aplikasi ini dikembangkan dengan menggunakan bahasa pemrograman python
            2. Library yang digunakan adalah streamlit, langchain, dan huggingface
            3. Aplikasi ini menggunakan Google Gemini API
            4. Aplikasi ini dikembangkan untuk tujuan Demo dan memiliki beberapa keterbatasan terkait limit/ quota yang digunakan dari Google Gemini API

        ## Cara Menggunakan Aplikasi "Chat with Website"
            1. Input URL Website yang akan menjadi referensi yang akan di proses Aplikasi menjadi sumber informasi, tekan enter atau klik tombol "Add" untuk menambahkan website referensi
            2. Kita dapat menambahkan beberapa url website sebagai sumber referensi dan informasi
            3. Apabila yang kita masukan bukan url yang valid maka akan muncul validasi pesan error, dan kita harus memasukan url website referensi yang valid
            4. Apabila kita sudah cukup memasukan satu atau beberapa url website sumber informasi, tampilan chat akan muncul di layar utama.
            5. Kita dapat menanyakan apapun pada Aplikasi "Chat with Website" selama informasi yang kita tanyakan ada pada website-website sumber informasi yang telah kita input.
        '''
    )
else:
    st.header("Chat with Websites 🌐",divider=True)

    user_message = st.chat_input()
    if user_message:
        st.session_state['messages'].append({
            'role': 'user',
            'message': user_message
        })

        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.write(message['message'])

        ai_answer = rag_response(user_prompt=user_message)

        st.session_state['messages'].append({
            'role':'ai',
            'message':ai_answer,
        })

        with st.chat_message('ai'):
            st.write(ai_answer)



