"""Ingest repository into the database."""

import argparse

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


descr_str = (
    "Uses LangChain tools and HuggingFace embedding to build ChromaDB from directory."
)
parser = argparse.ArgumentParser(
    prog="Ingest repo for ChromaDB", description=descr_str, fromfile_prefix_chars="@"
)

parser.add_argument(
        "--rag_dir",
        action="store",
        type=str,
        default="../yoke/src/yoke",
        help="Path to directory ChromaDB will be built from.",
    )

parser.add_argument(
    "--chunksize",
    action="store",
    type=int,
    default=1000,
    help="Size of chunks to create for ChromaDB.",
)

parser.add_argument(
    "--chunkoverlap",
    action="store",
    type=int,
    default=200,
    help="Number of overlap tokens in chunks.",
)

parser.add_argument(
    "--sentence_model",
    action="store",
    type=str,
    default="all-MiniLM-L6-v2",
    help=(
        "Name of HuggingFace sentence-transformer to use for token embedding."
    ),
)

parser.add_argument(
    "--chromadir",
    action="store",
    type=str,
    default="../yoke/chroma_db",
    help=(
        "Directory to store ChromaDB for RAG use."
    ),
)


if __name__ == "__main__":
    args = parser.parse_args()

    # 1. Load all .py and .md files from the repository
    loader = DirectoryLoader(
        args.rag_dir,
        glob=["**/*.py",
            "**/*.md"],
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from the repository.")

    # 2. Split the documents into 1000-token chunks with 200 tokens of overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunksize,
        chunk_overlap=args.chunkoverlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embed the chunks with a local SentenceTransformer model
    # Note: Need to figure out what models are available locally
    # and how to load them
    embeddings = HuggingFaceEmbeddings(
        model_name=args.sentence_model,
        model_kwargs={"device": "cuda"},
    )

    # 4. Create a Chroma vector store from the chunks and embeddings
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=args.chromadir,
    )
    print(f"Ingestion complete. Chroma DB at {args.chromadir}")