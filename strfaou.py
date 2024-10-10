import streamlit as st  # Importer Streamlit
import os
import openai
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfWriter, PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = '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'

os.environ["OPENAI_API_KEY"] = openai.api_key

llm_name = "gpt-3.5-turbo"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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"

def load_db(file, chain_type, k):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

class cbfs:
    def __init__(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_files = []
        self.merged_pdf_file = "merged.pdf"  # Chemin par défaut pour le fichier PDF fusionné
        self.qa = None  # Initialiser self.qa à None

    def merge_pdfs(self, pdf_files):
        from PyPDF2 import PdfWriter, PdfReader
        
        pdf_writer = PdfWriter()
        for pdf_file in pdf_files:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
        with open(self.merged_pdf_file, 'wb') as out:
            pdf_writer.write(out)

    def convchain(self, query):
        if not query:
            return "Veuillez entrer une question."
        if not self.qa:  # Vérifier si self.qa est initialisé
            return "Veuillez charger le fichier PDF pour initialiser la base de données."
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        return self.answer

cb = cbfs()

# Interface utilisateur Streamlit
st.title("NetSol - Telecom Chatbot")

# Uploader le fichier PDF
uploaded_file = st.file_uploader("Téléchargez votre fichier PDF", type="pdf")

if uploaded_file is not None:
    cb.loaded_files.append(uploaded_file)
    cb.merge_pdfs(cb.loaded_files)
    cb.qa = load_db(cb.merged_pdf_file, "stuff", 4)  # Charger la base de données
    st.success("Base de données chargée.")

# Champ de texte pour poser des questions
user_input = st.text_input("Posez votre question ici...")

if st.button("Envoyer"):
    response = cb.convchain(user_input)
    st.text_area("Réponse", value=response, height=300)

# Afficher l'historique de la conversation
if cb.chat_history:
    st.subheader("Historique des conversations")
    for exchange in cb.chat_history:
        st.write(f"**Utilisateur :** {exchange[0]}")
        st.write(f"**ChatBot :** {exchange[1]}")