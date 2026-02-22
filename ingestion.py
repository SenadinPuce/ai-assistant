import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Directory containing PDF files to ingest
PDF_DIR = "./docs"

# Pinecone configuration
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Load all PDFs from the directory
loader = PyPDFDirectoryLoader(PDF_DIR)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)

# Initialise Pinecone client and ensure the index exists
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

vectorstore = PineconeVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME,
)

retriever = vectorstore.as_retriever()