import warnings 
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from PyPDF2 import PdfReader
import urllib.parse
# import google.generativeai as genai
import getpass

GEMINI_API_KEY = getpass.getpass('Gemini_API_key: ')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


def get_embedds(file_path, filename):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists('doc_embeddings'):
        print('searching for doc embedding')

    else:
        os.mkdir('doc_embeddings')

    if not os.path.exists('doc_embeddings/' + filename):
        print('not found in vectorstore, creating and loading....')
        reader = PdfReader(file_path)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(corpus)

        if len(chunks) >=10:
            chunks = chunks[:10]

        vectors = FAISS.from_texts(chunks, embeddings)
        vectors.save_local(f'doc_embeddings/{filename}')

    else:
        print('loading from vectorstore')
        vectors = FAISS.load_local(f'doc_embeddings/{filename}', embeddings=embeddings, allow_dangerous_deserialization=True)

    return vectors

def start_chat(file_path):
    if file_path[0] == '"':
        file_path = file_path[1:]
    if file_path[-1] == '"':
        file_path = file_path[:-1]
    file_name = file_path.split('\\')[-1]
    file_name[:-4]
                                
    vectors = get_embedds(file_path, file_name)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectors.as_retriever(), return_source_documents=True)

    while True:
        print('\n\n##############################')
        query = input('Your question to pdf ("exit" to end): ')

        if query.lower() == 'exit':
            print('Thanks for talking....\n Exiting....')
            break
        
        print('Generating.....')
        chat_history = []
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))

        print('Answer:\n', chat_history[-1][1])