"""Query the ollama LLM with RAG capabilities."""

from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# 1. Load the Chroma vector store from the local directory
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="../yoke/chroma_db",
)

# 2. Spin up the Ollama LLM
llm = Ollama(
    model="gemma3:12b-it-qat",
    temperature=0.1,
    max_tokens=512,
    n_ctx=4096,
)

# 3. Create a RetrievalQA chain with the vector store and LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

# 4. Query the LLM with a question
question = input("Ask a question about the Yoke repository: ")
result = qa.invoke({"query": question})
print("Answer: ", result["result"])
print("Source documents: ")
for doc in result["source_documents"]:
    print(doc.metadata["source"])
    print(doc.page_content)
print("Done.")