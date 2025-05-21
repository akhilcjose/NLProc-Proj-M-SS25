
# NLProc-Proj-M-SS25 – Team Neon

A modular Natural Language Processing (NLP) system for performing various tasks such as Question Answering (QA), Summarization, Multiple Choice QA (MCQ), and Text Classification using retrieval-augmented generation.

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

To build a retrieval-augmented NLP system where given a query, the system fetches relevant context and generates accurate, context-specific responses.

![alt text](https://github.com/akhilcjose/NLProc-Proj-M-SS25/blob/feature/spec_doc/image.png)

---

## 🔧 Setup Instructions

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Ensure Required Files are in Place**

- `generator.py` for prompt generation and answer synthesis.
- `retriever_module.py` for embedding documents and retrieving top-k chunks.
- `test_inputs.json` for evaluation/testing input.
- `evaluation.py` for printing test evaluations.

---

## 🧠 Modules
### 1. **Requirements** – `requirements.txt`

# Creating a simple README.md content for a project based on the given requirements.txt


# 📦 Requirements Overview

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



## 🛠️ Usage

This file is useful for:

- ✅ Testing and debugging the generation pipeline.
- 📊 Comparing outputs of different models.
- 📁 Validating context-aware responses.

To load and use this file in Python:

```python
import json

with open('test_inputs.json', 'r') as file:
    test_data = json.load(file)

for test_case in test_data:
    print("Q:", test_case["question"])
    print("A:", test_case["generated_answer"])
    print("Chunks:", test_case["retrieved_chunks"])
    print("-----")
```

---

## 📂 Related Files

- `generator.py`: Builds prompts and generates answers using a transformer model.
- `retriever_module.py`: Retrieves relevant chunks for a given query.
- `requirements.txt`: Lists required dependencies.

---


### 2. **Retriever** – `retriever_module.py`

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

```python
chunk_document(text, chunk_size=20, overlap=10, method='fixed')
```

- ✅ Simple, good for short texts  
- ❌ Can break sentences and lose semantic meaning

---

### 2. 🧱 Sentence-based Chunking

Splits the document by sentences and groups a fixed number of them per chunk.

```python
chunk_document(text, num_sentences=3, method='sentence')
```

- ✅ Maintains grammatical meaning  
- ❌ Uneven lengths, may exceed token limits

---

### 3. 📏 Paragraph-based Chunking

Divides the document by paragraphs (`\n\n` as delimiter).

```python
chunk_document(text, method='paragraph')
```

- ✅ Keeps logical structure intact  
- ❌ Paragraphs can be too long or too short

---

### 4. 🔢 Token-based Chunking (Advanced)

Uses a tokenizer (e.g., from HuggingFace) to split text into chunks based on token limits.

```python
chunk_document(text, max_tokens=128, method='token')
```

- ✅ Optimized for LLM input sizes  
- ❌ Requires external tokenizer and handling edge cases

---

## ❓ Why Chunking?

> Large documents often exceed model input limits. Chunking divides them into digestible segments while preserving enough context for relevant retrieval.

---

## 🧱 Class Structure

```python
class Retriever:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2')
    def chunk_document(self, document, method='fixed', ...)
    def add_documents(self, documents)
    def query(self, query_text, top_k=3)
    def save(self)
    def load(self)
```

---

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

## 🧪 Example

```python
retriever = Retriever()
retriever.add_documents(["Natural language processing is a branch of AI..."])
results = retriever.query("What is NLP?")
print(results)
```



### 3 **Generator** – `generator.py`

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

## 💡 Example Usage

### ✅ Example: Question Answering (`qa`)

```python
from generator import Generator

gen = Generator()
retrieved_chunks = [{"text": "The moon is Earth's only natural satellite."}]
question = "What is the moon?"

answer = gen.generate_answer(task="qa", question=question, retrieved_chunks=retrieved_chunks)
print("Answer:", answer)
```

### 📝 Example: Summarization

```python
context = [{"text": "Artificial Intelligence enables machines to mimic human intelligence. It includes learning and problem-solving."}]
gen = Generator()

summary = gen.generate_answer(task="summarization", retrieved_chunks=context)
print("Summary:", summary)
```

### ❓ Example: Multiple Choice (`mcq`)

```python
question = "Which planet is known as the Red Planet?"
context = [{"text": "Mars is often called the Red Planet due to its reddish appearance."}]
options = ["Earth", "Mars", "Venus", "Jupiter"]

answer = gen.generate_answer(task="mcq", question=question, retrieved_chunks=context, options=options)
print("MCQ Answer:", answer)
```

### 🚨 Example: Classification

```python
context = [{"text": "Offensive: Uses hurtful or abusive language.\nNon-Offensive: Respectful and neutral language."}]
text_to_classify = "You are a fool!"

label = gen.generate_answer(task="classification", retrieved_chunks=context, text_to_classify=text_to_classify)
print("Label:", label)
```

---

## 🧪 Notes

- The model uses beam search (`num_beams=4`) for more robust outputs.
- `max_new_tokens` is capped at 100 to keep responses concise.
- If context is missing or irrelevant, it responds with defaults like “I don't know.”

---

## 📦 Dependencies

- `transformers`
- `torch`

Make sure to install them via:
```bash
pip install transformers torch
```

---

## 📁 File: generator.py

Make sure your directory structure includes `generator.py` and the examples above to test different task types.


### 4. **Evaluator** – `evaluation.py`

Loads `test_inputs.json` and prints question, retrieved context, generated answer, and metadata.

Use this to verify your implementation:

```bash
python baseline/evaluation.py
```

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

## 🧰 Tools & Libraries

- `transformers` – For sequence generation (`flan-t5`)
- `sentence-transformers` – For semantic embeddings
- `faiss-cpu` – For fast similarity search
- `torch` – Backend for model inference

---

## 👥 Team Neon

Created as part of the SS25 NLP project module.

---

## 📌 Notes

- The system is modular and task-specific.
- Easy to extend with new models or embedding techniques.
- Can be integrated into larger QA or chatbot systems.

---

## 📷 Suggested Improvements


---

