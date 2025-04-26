Repo RAG - Learn Your Code
==========================

This is a bare-bones set of instructions an scripts to set up a local RAG
system for a local repository for purposes of interactive code documentation.

Requirements and Dependencies:
------------------------------

- Ubuntu
- Python
- Ollama

Instructions:
-------------

```
>> apt update
>> apt install -y curl
>> curl -fsSL https://ollama.com/install.sh | sh
```

Once installed, confirm...

```
>> ollama --version
>> ollama --help
```

Pull a Model
------------

Ollama bundles model weights and runtime, your just pull the model you need:

```
>> ollama pull mistral
>> ollama pull gemma3:12b-it-qat
```

(*You can find a list of models at https://ollama.com/models*)

You can test that the model pulled with the following:

```
>> ollama list
>> ollama show mistral
>> ollama run mistral "Can you tell me about RAG please?"
```


Create python environment:
--------------------------

We use anaconda.

```
>> conda create -n rag_env python=3.10 pip
>> conda activate rag_env
>> pip install --upgrade pip
>> pip install torch torchvision torchaudio
>> pip install langchain chromadb sentence-transformers ollama
>> pip install langchain-community langchain-huggingface
>> pip install unstructured
```

- **langchain**: orchestration
- **chromadb**: fast local vector store
- **sentence-transformers**: embedding model
- **ollama**: python bindings for ollama CLI

Build the RAG database:
-----------------------

```
>> python ingest.py
```

Query the LLM with the database:
--------------------------------

```
>> python query.py
```