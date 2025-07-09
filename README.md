
# NLProc-Proj-M-SS25 – Team Neon

This project is designed to assist users in querying academic research papers and retrieving relevant, context-specific answers. The system focuses on understanding the user's query and extracting accurate information directly from the content of the given paper, enabling more efficient and insightful academic research.

---

## 📂 Project Structure

```
NLProc-Proj-M-SS25/
├── generator/
│   └── generator.py              # Generator class: build_prompt(), generate_answer()
├── retriever/
│   └── retriever.py              # Retriever class: add_documents(), query(), save(), load()
├── evaluation/
│   └── evaluation.py             # Logging, test runs, grounding checks
├── pipeline.py 
├── test_inputs.json              # Known Q&A pairs for testing
├── requirements.txt              # Project dependencies
└── README.md                     # Project overview and instructions

```

---

## 🎯 Objective

To build a retrieval-augmented NLP system that takes a user query along with a research paper as input and returns precise answers from the paper’s content. This supports quick knowledge extraction and deeper understanding of scholarly texts.




---

## 🚀 Features

- Post natural language queries.
- Retrieve relevant answers from research paper.
- Improve research efficiency through semantic search.

---

## 🧠 Modules



# 1. **Retriever** – `retriever_module.py`

The `Retriever` class provides a modular interface for building a semantic retriever using **SentenceTransformers** for embeddings and **FAISS** for fast similarity search. It is designed for tasks like Question Answering (QA), search, and context retrieval in Retrieval-Augmented Generation (RAG) systems.

---

## 🔧 Features

- ✅ Multiple document chunking strategies  
- ✅ Embedding using `SentenceTransformer`  
- ✅ FAISS index creation and querying  
- ✅ Top-k similarity-based retrieval  

---

## 🧠 Core Components

- **_init__** - Initializes the retriever with a SentenceTransformer model and sets up data structures for storing documents and embeddings.
- **chunk_document** -  Splits a document into overlapping word chunks to preserve context for embedding and retrieval.
- **add_documents** - Processes and embeds input documents, then builds a FAISS index for efficient similarity search.
- **query** - Finds and returns the most relevant document chunks based on the semantic similarity of the input query.
- **load** - Placeholder method intended for implementing future model and index loading capabilities.

---

## 📚 Chunking Strategy

### 1. 🧱 Semantic Section-based Chunking (Used in this project)

Splits the document first by semantic section headers (e.g., numbered headings like `4.2 Results`) and then applies recursive character-based splitting to large sections.

- ✅ Preserves document structure and section semantics  
- ✅ Reduces the risk of breaking meaningful content mid-way  
- ❌ Requires documents with clear section headers for best results

---

### 2. ✂️ Recursive Character-based Chunking (Fallback)

Uses LangChain's `RecursiveCharacterTextSplitter` to break down text based on character limits and common separators (`\n\n`, `\n`, `.`, etc.).

- ✅ Ensures manageable chunk sizes for embedding models  
- ✅ Works well even without clear section headers  
- ❌ Can break logical units of meaning if not tuned carefully

---

## 🔍 Retrieval Pipeline

### Overview

This pipeline retrieves the most relevant document chunks in response to a user query using a hybrid of dense retrieval and re-ranking.

---

### 1. 🧠 Dense Vector Retrieval (FAISS + SentenceTransformer)

Generates embeddings for document chunks using a SentenceTransformer model (`paraphrase-MiniLM-L6-v2` by default), and indexes them using FAISS for efficient similarity search.

- ✅ Fast and scalable retrieval  
- ✅ Works well for semantically similar content  
- ❌ Might return loosely related chunks without re-ranking

---

### 2. 🎯 Cross-Encoder Re-Ranking (Optional)

The top-k retrieved chunks are re-scored using a more accurate cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which jointly encodes the query and document chunk.

- ✅ Improves relevance of final results  
- ✅ Learns fine-grained matching  
- ❌ Slower than dense retrieval (only used on top results)

---

### 3. 🔄 Retrieval Workflow

1. Input query is encoded into a dense vector  
2. FAISS searches top `k` similar chunks  
3. Top chunks are re-ranked by the cross-encoder  
4. Final top `rerank_k` chunks are returned





---

## 🖼️ Architecture Diagram

```
[Document]
     ↓
[Chunking Strategy]
     ↓
[Embedding (SentenceTransformers)]
     ↓
[FAISS Index]
     ↑
[Query Embedding] ← [Query]
     ↓
[Top-k Similar Chunks]
```

![alt text](https://github.com/akhilcjose/NLProc-Proj-M-SS25/blob/feature/spec_doc/system%20architecture.png)
---



# 2. **Generator** – `generator.py`

The `generator.py` module is responsible for generating textual responses based questions. It uses a pre-trained transformer model (google/flan-t5-large) from the Hugging Face Transformers library.

## 🧠 How It Works

### Core Components:
- **Tokenizer & Model Initialization**: Loads the pre-trained model and tokenizer.
- **Prompt Builder**: Constructs task-specific prompts using the context (retrieved document chunks, question, etc.)
- **Answer Generator**: Uses beam search to generate a concise response.


---

## 📝 Prompt Construction Logic

- **Question Answering (QA):**
    - Requires context and question.
    - Ensures the model only answers if the context contains the answer.



---


# 3. **Evaluator** – `evaluation.py`

Loads `test_inputs.json` and prints question, retrieved context, generated answer, and metadata.


---
# 4. **Requirements** – `requirements.txt`

## 📦 Requirements Overview

This repository contains the dependencies needed for a Natural Language Processing (NLP) project that utilizes document retrieval and transformer-based embeddings.

## 📚 Included Libraries



## 📦 Dependencies

This project relies on a set of powerful libraries for document processing, semantic retrieval, and web deployment.

### 🔍 NLP & Embedding
- `sentence-transformers` – For dense vector embedding of text chunks  
- `faiss-cpu` – Fast approximate nearest neighbor search  
- `transformers` – For cross-encoder re-ranking  
- `langchain`, `langchain-core`, `langchain-text-splitters` – For chunking and document pipeline management  

### 📄 Document Parsing
- `pymupdf`, `pypdf2`, `pypdfium2` – PDF parsing  
- `python-docx`, `python-pptx`, `openpyxl`, `xlsxwriter` – DOCX, PPTX, and Excel parsing  
- `beautifulsoup4`, `lxml` – HTML/XML parsing  

### 📚 Semantic Chunking
- `semchunk` – For section-aware document chunking  
- `spacy`, `spacy-layout` – For layout-based sentence segmentation  

### ⚙️ Utilities & Backend
- `fastapi`, `uvicorn` – Web API interface  
- `gradio` – Interactive UI (optional)  
- `pandas`, `numpy`, `scikit-learn`, `scipy` – Data handling and evaluation  
- `joblib`, `multiprocess`, `mpire` – Parallel processing  
- `python-dotenv`, `pydantic` – Config and environment management  


---

## 👥 Team Neon

Created as part of the SS25 NLP project module.

---

## 📌 Notes

- The system is modular and task-specific.
- Easy to extend with new models or embedding techniques.
- Can be integrated into larger QA or chatbot systems.


---

