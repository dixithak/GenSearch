import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from gensearch.config import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import AnthropicLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


def get_pdf_text(docs):
    text =""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    # Split the text into chunks, chuinks of 1000 characters with 20 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    #initialize the embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)

    #create vectorestore from embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def conversational_chain(vectorstore, model_key = 'GEMINI_2_5_PRO'):
    model = get_model_api_name(model_key=model_key)
    #initialize the LLM
    if model.startswith("gemini"):
        llm = GoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0)
    elif model.startswith("claude"):
        llm = AnthropicLLM(model=model, anthropic_api_key = ANTHROPIC_API_KEY,temperature=0)

    # To retain conversation history and previous context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the Conversational Retrieval Chain with llm, vectorstore and memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain