"""Provide simple WebUI for RepoRAG."""

# Prevent streamlit from trying to wal torch's C++ classes
import torch
torch.classes.__path__ = []

import argparse
import streamlit as st
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate


def parse_args():
    descr_str = (
        "Uses Ollama LLM, LangChain, and HuggingFace embedding "
        "to chat with a local RAG LLM."
    )
    parser = argparse.ArgumentParser(
        prog="Local RAG LLM Chat", description=descr_str, fromfile_prefix_chars="@"
    )

    parser.add_argument(
            "--rag_dir",
            action="store",
            type=str,
            default="../yoke/src/yoke",
            help="Path to directory ChromaDB will be built from.",
        )

    parser.add_argument(
        "--llm_model",
        action="store",
        type=str,
        default="gemma3:12b-it-qat",
        help=(
            "Name of Ollama LLM to converse with."
        ),
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

    # Only parse known args so streamlit arguments prior to -- are not interpreted
    # as an error
    args, _ = parser.parse_known_args()

    return args


def run_repo_rag(
        sentence_model: str,
        llm_model: str,
        chromadir: str,
        ):
    """Streamlit RAG chat. 
    
    Function to start a chat with an Ollama LLM with access to a Chroma DB.

    Args:
        sentence_model (str): Name of hugging-face sentence embedding model.
        llm_model (str): Name of Ollama local LLM
        chromadir (str): Path to Chroma DB for RAG.

    """

    # 1) Load you DB and models
    @st.cache_resource
    def load_components():
        # Embeddings and vector store
        embedding = HuggingFaceEmbeddings(
            model_name=sentence_model,
            model_kwargs={"device": "cuda"},
        )
        vectordb = Chroma(
            persist_directory=chromadir,
            embedding_function=embedding
        )

        # LLM via Ollama
        llm = Ollama(
            model=llm_model,
            temperature=0.1,
            max_tokens=512,
            n_ctx=4096,
        )

        # Set up in-memory chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Feed a prompt to instruct use of conversation history.
        condense_question_prompt = PromptTemplate.from_template(
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question.\n\n"
            "Chat History:\n{chat_history}\n"
            "Follow Up Input: {question}\n"
            "Standalone question:"
        )

        # Set up a prompt to allow LLM to fallback on previous knowledge
        combine_documents_prompt = PromptTemplate.from_template(
            """You are an assistant that knows your local codebase *and* general world
            facts.

              Context snippets from codebase (if any):
              {context}
  
            Answer the question: {question}

            Use a synthesis of your own knowledge and the context to answer the
            question.
            """
        )

        # Build the conversational RAG chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": combine_documents_prompt},
            return_source_documents=False,
        )

        return qa

    qa_chain = load_components()

    st.title("RepoRAG Chat")

    # 2) Initialize history in session store
    if "history" not in st.session_state:
        # history will be a list of {"q": ..., "a": ...}
        st.session_state.history = []

    # 3) Display the chat history
    for entry in st.session_state.history:
        st.markdown(f"**You:** {entry['q']}")
        st.markdown(f"**[{llm_model}] RepoRAG-AI: {entry['a']}")
        st.write("---")

    # 4) Input box + submit handler
    def handle_submit():
        query = st.session_state.query_input.strip()
        if not query:
            return

        with st.spinner("Thinking..."):
            # Now invoke the conversation chain: it returns 
            # {"answer": ..., "chat_history": ...}
            response = qa_chain.invoke({"question": query})
        answer = response["answer"]

        # store in history
        st.session_state.history.append({"q": query, "a": answer})
        # clear input
        st.session_state.query_input = ""

    st.text_input(
        "Ask me about your repo:",
        key="query_input",
        on_change=handle_submit
    )


if __name__ == "__main__":
    args = parse_args()

    run_repo_rag(
        sentence_model=args.sentence_model,
        llm_model=args.llm_model,
        chromadir=args.chromadir,
        )