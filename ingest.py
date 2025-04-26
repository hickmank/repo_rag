"""Ingest repository into the database."""

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load all .py and .md files from the repository
loader = DirectoryLoader(
    "../yoke/src/yoke",
    glob=["**/*.py",
          "**/*.md"],
)
docs = loader.load()
print(f"Loaded {len(docs)} documents from the repository.")

# 2. Split the documents into 1000-token chunks with 200 tokens of overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# 3. Embed the chunks with a local SentenceTransformer model
# Note: Need to figure out what models are available locally
# and how to load them
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)

# 4. Create a Chroma vector store from the chunks and embeddings
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="../yoke/chroma_db",
)
print("Ingestion complete. Chroma DB on ./yoke/chroma_db")