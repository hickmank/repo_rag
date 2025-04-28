"""Provide simple WebUI for RepoRAG."""

# Prevent streamlit from trying to wal torch's C++ classes
import torch
torch.classes.__path__ = []

import streamlit as st
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# To turn on conversational memory toggle the following.
# NOTE: Under Construction! Expect errors...
CONVERSATIONAL = True

# 1) Load you DB and models
@st.cache_resource
def load_components():
    # Embeddings and vector store
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )
    vectordb = Chroma(
        persist_directory="../yoke/chroma_db",
        embedding_function=embedding
    )

    # LLM via Ollama
    llm = Ollama(
        model="gemma3:12b-it-qat",
        temperature=0.1,
        max_tokens=512,
        n_ctx=4096,
    )

    # Set up in-memory chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    if CONVERSATIONAL:
        # Build the conversational RAG chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=False
        )
    else:
        # QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
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
    st.markdown(f"**RepoRAG-AI: {entry['a']}")
    st.write("---")

# 4) Input box + submit handler
def handle_submit():
    query = st.session_state.query_input.strip()
    if not query:
        return
    if CONVERSATIONAL:
        with st.spinner("Thinking..."):
            # Now invoke the conversation chain: it returns 
            # {"answer": ..., "chat_history": ...}
            response = qa_chain.invoke({"question": query})
        answer = response["answer"]
    else:
        with st.spinner("Thinking..."):
            # invoke() returns a dict; "result" holds the answer
            response = qa_chain.invoke({"query": query})
        answer = response["result"]

    # store in history
    st.session_state.history.append({"q": query, "a": answer})
    # clear input
    st.session_state.query_input = ""

st.text_input(
    "Ask me about your repo:",
    key="query_input",
    on_change=handle_submit
)
