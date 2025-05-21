
# NLProc-Proj-M-SS25 – Team Neon

This project is designed to assist users in retrieving relevant research papers based on their queries. It aims to simplify and accelerate the process of academic research by providing intelligent, context-aware search results in response to natural language input.

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
├── test_inputs.json              # Known Q&A pairs for testing
├── requirements.txt              # Project dependencies
└── README.md                     # Project overview and instructions

```

---

## 🎯 Objective

To build a retrieval-augmented NLP system where, given a query, the system fetches relevant research context and generates accurate, context-specific responses.

![alt text](https://github.com/akhilcjose/NLProc-Proj-M-SS25/blob/feature/spec_doc/image.png)


---

## 🚀 Features

- Post natural language queries.
- Retrieve relevant research paper results.
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

### `chunk_document(document, method='fixed', ...)`

Splits a document into smaller chunks using one of several strategies.

---

## 📚 Chunking Strategies

### 1. ✂️ Fixed-size Overlapping Windows (Default)

Splits text into word-based chunks with overlapping words between them.

- ✅ Simple, good for short texts  
- ❌ Can break sentences and lose semantic meaning

---

### 2. 🧱 Sentence-based Chunking

Splits the document by sentences and groups a fixed number of them per chunk.

- ✅ Maintains grammatical meaning  
- ❌ Uneven lengths, may exceed token limits

---

### 3. 📏 Paragraph-based Chunking

Divides the document by paragraphs (`\n\n` as delimiter).

- ✅ Keeps logical structure intact  
- ❌ Paragraphs can be too long or too short

---

### 4. 🔢 Token-based Chunking (Advanced)

Uses a tokenizer (e.g., from HuggingFace) to split text into chunks based on token limits.

- ✅ Optimized for LLM input sizes  
- ❌ Requires external tokenizer and handling edge cases

---

## ❓ Why Chunking?

> Large documents often exceed model input limits. Chunking divides them into digestible segments while preserving enough context for relevant retrieval.



## 🔁 Other Methods

- **`add_documents(documents)`**: Splits, embeds, and indexes documents  
- **`query(query_text, top_k=3)`**: Returns most relevant chunks  
- **`save()` / `load()`**: Save or load FAISS index and embeddings

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

---



# 2 **Generator** – `generator.py`

The `generator.py` module is responsible for generating textual responses based on tasks like Question Answering (QA), Summarization, Multiple-Choice Question (MCQ) answering, and Text Classification. It uses a pre-trained transformer model (default: `google/flan-t5-base`) from the Hugging Face Transformers library.

## 🧠 How It Works

### Main Components:
- **Tokenizer & Model Initialization**: Loads the pre-trained model and tokenizer.
- **Prompt Builder**: Constructs task-specific prompts using the context (retrieved document chunks, question, etc.)
- **Answer Generator**: Uses beam search to generate a concise response.

## 🔧 Tasks Supported

| Task            | Input Parameters                               | Output                  |
|-----------------|------------------------------------------------|--------------------------|
| QA              | `question`, `retrieved_chunks`                | One-sentence answer     |
| Summarization   | `retrieved_chunks`                            | One-sentence summary    |
| MCQ             | `question`, `retrieved_chunks`, `options`     | One letter + option     |
| Classification  | `text_to_classify`, `retrieved_chunks`        | "Offensive" or "Non-Offensive" |

---

## 📝 Prompt Construction Logic

- **Question Answering (QA):**
    - Requires context and question.
    - Ensures the model only answers if the context contains the answer.

- **Summarization:**
    - Summarizes content into one sentence without using external knowledge.

- **MCQ:**
    - Generates the best answer from given options using only the context.

- **Classification:**
    - Classifies a text based on the definitions of "Offensive" and "Non-Offensive" given in the context.


---


# 3. **Evaluator** – `evaluation.py`

Loads `test_inputs.json` and prints question, retrieved context, generated answer, and metadata.

---

## 🧪 Sample Data – `test_inputs.json`

```json
{
  "question": "What is natural language processing?",
  "retrieved_chunks": ["NLP is a subfield of AI ..."],
  "generated_answer": "Natural language processing ...",
  "group_id": "Team_Neon"
}
```

---
# 4. **Requirements** – `requirements.txt`

## 📦 Requirements Overview

This repository contains the dependencies needed for a Natural Language Processing (NLP) project that utilizes document retrieval and transformer-based embeddings.

## 📚 Included Libraries

### 1. **faiss-cpu**
- A library for efficient similarity search and clustering of dense vectors.
- Used for fast retrieval of document chunks using approximate nearest neighbor search.

### 2. **sentence-transformers**
- Framework for generating sentence and text embeddings using pretrained models like BERT, RoBERTa, etc.
- Essential for encoding documents and queries into vectors for semantic search.

### 3. **numpy**
- Fundamental package for numerical computation.
- Used for handling embeddings and matrix operations required by FAISS.

### 4. **PyPDF2**
- Pure Python library to read and extract text from PDF files.
- Useful for loading real documents in `.pdf` format for processing.



---

## 👥 Team Neon

Created as part of the SS25 NLP project module.

---

## 📌 Notes

- The system is modular and task-specific.
- Easy to extend with new models or embedding techniques.
- Can be integrated into larger QA or chatbot systems.


---

