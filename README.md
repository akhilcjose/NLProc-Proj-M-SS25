
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
├── test_inputs.json              # Known Q&A pairs for testing
├── requirements.txt              # Project dependencies
└── README.md                     # Project overview and instructions

```

---

## 🎯 Objective

To build a retrieval-augmented NLP system that takes a user query along with a research paper as input and returns precise answers from the paper’s content. This supports quick knowledge extraction and deeper understanding of scholarly texts.

![alt text]([https://github.com/akhilcjose/NLProc-Proj-M-SS25/blob/feature/spec_doc/system%20architecture.png])


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



# 2. **Generator** – `generator.py`

The `generator.py` module is responsible for generating textual responses based on tasks like Question Answering (QA), Summarization, Multiple-Choice Question (MCQ) answering, and Text Classification. It uses a pre-trained transformer model (default: `google/flan-t5-base`) from the Hugging Face Transformers library.

## 🧠 How It Works

### Core Components:
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

