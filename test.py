from CustomLLM import EYQIncubator
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.laser import LaserEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
# from langchain.document_loaders import DirectoryLoader
import sys
import os
import logging
#from utilities import pretty_prints

logging.basicConfig(level=logging.INFO, filename="langchain.log")

end_point = os.getenv("EYQ_INCUBATOR_ENDPOINT")
x_api_key = os.getenv("EYQ_INCUBATOR_KEY")


llm = EYQIncubator(end_point=end_point,
                       x_api_key=x_api_key,
                       model="gpt-4-turbo",
                       api_version="2023-05-15")


# loader = DirectoryLoader("./documents/",glob="./*.pdf",loader_cls=PyMuPDFLoader)
loader = PyMuPDFLoader("resources/tmp/ISTQB_CTFL_Syllabus-v4.0.pdf")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)

docs = text_splitter.split_documents(loader.load())

persist_directory = "db"

vector_store = Chroma.from_documents(documents=docs,embedding= LaserEmbeddings(),persist_directory=persist_directory)
# vector_store = Chroma(persist_directory="db",embedding_function=LaserEmbeddings())

retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":1})

qa_chain = RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever,return_source_documents=True)

try:
    while True:
        user_query = input("[Question]: ")
        answer = qa_chain.invoke(user_query)
       # pretty_prints(answer["result"])
        print("\n\nSources:")
        for source in answer["source_documents"]:
            print(source.metadata['source'])
        print()
except KeyboardInterrupt:
    print("\n\n[!] Exiting...\n")
    sys.exit(0)