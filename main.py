from CustomLLM import EYQIncubator
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.laser import LaserEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, filename="langchain.log")

# end_point = os.getenv("EYQ_INCUBATOR_ENDPOINT")
# x_api_key = os.getenv("EYQ_INCUBATOR_KEY")
end_point = 'https://eyqincubator.america.fabric.ey.com/eyq/canadaeast/api'
x_api_key = 'DI0xDSceUvmWFmAwEaWCsXM3SoyHDwWE'


llm = EYQIncubator(end_point=end_point,
                       x_api_key=x_api_key,
                       model="gpt-4-turbo",
                       api_version="2023-05-15")


template = """
Answer the question based on the context below. If you don't know the answer or can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

loader = PyPDFLoader("resources/tmp/ISTQB_CTFL_Syllabus-v4.0.pdf")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)

docs = text_splitter.split_documents(loader.load())

vector_store = Chroma.from_documents(docs, LaserEmbeddings())

retriever = vector_store.as_retriever()


chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt
         | llm
         | output_parser)

try:
    while True:
        user_query = input("[Question]: ")
        answer = chain.invoke(user_query)
        print(answer)
except KeyboardInterrupt:
    print("\n\n[!] Exiting...\n")
    sys.exit(0)