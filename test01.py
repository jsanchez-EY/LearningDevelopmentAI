from langchain.chains.retrieval_qa.base import RetrievalQA

from CustomLLM import EYQIncubator
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.laser import LaserEmbeddings
from langchain_chroma import Chroma
import sys
import logging
import os


logging.basicConfig(level=logging.INFO, filename="langchain.log")

# end_point = os.getenv("EYQ_INCUBATOR_ENDPOINT")
# x_api_key = os.getenv("EYQ_INCUBATOR_KEY")

end_point = 'https://eyqincubator.america.fabric.ey.com/eyq/canadaeast/api'
x_api_key = 'DI0xDSceUvmWFmAwEaWCsXM3SoyHDwWE'


llm = EYQIncubator(end_point=end_point,
                       x_api_key=x_api_key,
                       model="gpt-4-turbo",
                       api_version="2023-05-15")

# loader = PyMuPDFLoader("resources/ISTQB_CTFL_Syllabus-v4.0.pdf")
loader = DirectoryLoader('./resources/', glob='./*.pdf', loader_cls=PyMuPDFLoader)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
docs = text_splitter.split_documents(loader.load())

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_function = LaserEmbeddings()
# embedding_function = HuggingFaceEmbeddings()

# load it into Chroma
vector_store = Chroma.from_documents(docs, embedding_function)

retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":1})

qa_chain = RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever,return_source_documents=True)

try:
    while True:
        user_query = input("[Question]: ")
        docs = vector_store.similarity_search(user_query)
        answer = qa_chain.invoke(user_query)
        # print(docs[0].page_content)
        print(answer["result"])
        print("\n\nSources:")
        for source in answer["source_documents"]:
            print(source.metadata['source'])
        print()
except KeyboardInterrupt:
    print("\n\n[!] Exiting...\n")
    sys.exit(0)