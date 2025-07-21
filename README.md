
# NLProc-Proj-M-SS25 – Team Neon

This project is designed to assist users in querying academic research papers and retrieving relevant, context-specific answers. The system focuses on understanding the user's query and extracting accurate information directly from the content of the given paper, enabling more efficient and insightful academic research.It uses techniques like semantic search and language generation to understand your question and find the best answer from the paper.

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

Loads `test_inputs01.json` runs through the system, then prints the question,retrieved content and the generated answer.

---

## 🧪 Sample Log file – `log_with_bertscore.json`

```json
  {
    "group_id": "Team_Neon",
    "question": "What is the primary goal of the GdVAE model?",
    "expected_answer": "The GdVAE model aims to unify self-explainable models and counterfactual explanations by using a conditional variational autoencoder with a Gaussian discriminant analysis classifier.",
    "generated_answer": "transform the features z from the recognition network and marginalization process into an interpretable class prediction",
    "generated_promt": "Answering the question using only the provided context.\nIf the answer is not in the context, respond with 'I don't know.'\nContext:\n1\nEvaluation of Predictive Performance\nMethodology. For a trustworthy SEM, performance should align with the clos-\nest black-box model [13]. Thus, the goal of this evaluation is not to outperform\nstate-of-the-art results on specific datasets but to offer a relative comparison for\nthe GdVAE architecture and various training methods. In all approaches, both\nthe classifier and autoencoder are jointly trained, sharing the same backbone.\nBaselines. First, optimal performance for the selected architecture is established\nusing a black-box model, comprising a jointly trained CVAE and classifier as\nthe baseline. Next, GdVAE\u2019s inference method is evaluated against the leading\nCVAE technique, importance sampling (IS) [45,48,54]. Lastly, ProtoVAE [13] is\nreferenced as a prototype-per-class VAE benchmark.\nResults. The results in Tab. 2 indicate good generalization in classification and\nreconstruction across MNIST and CelebA. The GdVAE\u2019s EM-based inference\nfor higher-dimensional images, benefiting from sampling in the lower-dimensional\nlatent instead of image space. With data augmentation and normalization from\nProtoVAE, GdVAE achieves comparable results to ProtoVAE.\nTakeaway: The inference procedure of our SEM closely matches the perfor-\nmance of a discriminative black-box model. Furthermore, our method consis-\ntently delivers competitive results to state-of-the-art approaches, particularly\nwhen applied to higher-dimensional images. The class-conditional GdVAE offers\nbetter reconstructions compared to ProtoVAE, the only unconditional model.\nGdVAE is shown in through easily comprehensible global explanations and latent space visualization.\nWe achieve this by displaying the decoded prototypes and interpolating between\nthem through our global explainer function (see fier\u2019s prototypes directly uncover biases without the need for quantitative anal-\nysis of counterfactuals on simulated datasets, as shown in prior work (e.g., [43]).\ncriminant analysis model (GDA) [18] and does not have any additional parame-\nters. Its purpose is to transform the features z from the recognition network and\nmarginalization process into an interpretable class prediction.\nDuring the training of the entire GdVAE, the prior network learns the class-\nconditional mean \u00b5z(y; \u03b8) = \u00b5z|y and covariance \u03a3z(y; \u03b8) = \u03a3z|y as the parame-\nters of our distribution p\u03b8(z|y) = N (\u00b5z(y; \u03b8), \u03a3z(y; \u03b8)). We assume conditional\nindependence and decompose the likelihood as p\u03b8(z|y) = QM\nj=1 p\u03b8(zj|y). In prac-\ntice, this results in a diagonal covariance matrix \u03a3z|y = diag\n\u0010\n\u03c32\nz1|y, . . . , \u03c32\nzM|y\n\u0011\n.\nWe use this distribution to determine the likelihood values for the GDA classi-\nfier. The class prior p\u03b8(y) can be learned either jointly or separately as the final\ncomponent of the GDA model. Thus, we use the mean values as class prototypes\nand the covariance to measure the distance to these prototypes.\n\nQuestion: What is the primary goal of the GdVAE model?\nAnswer:",
    "retrieved_chunks": [
      {
        "text": "1\nEvaluation of Predictive Performance\nMethodology. For a trustworthy SEM, performance should align with the clos-\nest black-box model [13]. Thus, the goal of this evaluation is not to outperform\nstate-of-the-art results on specific datasets but to offer a relative comparison for\nthe GdVAE architecture and various training methods. In all approaches, both\nthe classifier and autoencoder are jointly trained, sharing the same backbone.\nBaselines. First, optimal performance for the selected architecture is established\nusing a black-box model, comprising a jointly trained CVAE and classifier as\nthe baseline. Next, GdVAE\u2019s inference method is evaluated against the leading\nCVAE technique, importance sampling (IS) [45,48,54]. Lastly, ProtoVAE [13] is\nreferenced as a prototype-per-class VAE benchmark.\nResults. The results in Tab. 2 indicate good generalization in classification and\nreconstruction across MNIST and CelebA. The GdVAE\u2019s EM-based inference",
        "score": 4.303585052490234
      },
      {
        "text": "for higher-dimensional images, benefiting from sampling in the lower-dimensional\nlatent instead of image space. With data augmentation and normalization from\nProtoVAE, GdVAE achieves comparable results to ProtoVAE.\nTakeaway: The inference procedure of our SEM closely matches the perfor-\nmance of a discriminative black-box model. Furthermore, our method consis-\ntently delivers competitive results to state-of-the-art approaches, particularly\nwhen applied to higher-dimensional images. The class-conditional GdVAE offers\nbetter reconstructions compared to ProtoVAE, the only unconditional model.",
        "score": 2.6142544746398926
      },
      {
        "text": "GdVAE is shown in through easily comprehensible global explanations and latent space visualization.\nWe achieve this by displaying the decoded prototypes and interpolating between\nthem through our global explainer function (see fier\u2019s prototypes directly uncover biases without the need for quantitative anal-\nysis of counterfactuals on simulated datasets, as shown in prior work (e.g., [43]).",
        "score": 2.271634578704834
      },
      {
        "text": "criminant analysis model (GDA) [18] and does not have any additional parame-\nters. Its purpose is to transform the features z from the recognition network and\nmarginalization process into an interpretable class prediction.\nDuring the training of the entire GdVAE, the prior network learns the class-\nconditional mean \u00b5z(y; \u03b8) = \u00b5z|y and covariance \u03a3z(y; \u03b8) = \u03a3z|y as the parame-\nters of our distribution p\u03b8(z|y) = N (\u00b5z(y; \u03b8), \u03a3z(y; \u03b8)). We assume conditional\nindependence and decompose the likelihood as p\u03b8(z|y) = QM\nj=1 p\u03b8(zj|y). In prac-\ntice, this results in a diagonal covariance matrix \u03a3z|y = diag\n\u0010\n\u03c32\nz1|y, . . . , \u03c32\nzM|y\n\u0011\n.\nWe use this distribution to determine the likelihood values for the GDA classi-\nfier. The class prior p\u03b8(y) can be learned either jointly or separately as the final\ncomponent of the GDA model. Thus, we use the mean values as class prototypes\nand the covariance to measure the distance to these prototypes.",
        "score": 2.063140869140625
      }
    ],
    "bert_score_precision": 0.847,
    "bert_score_recall": 0.828,
    "bert_score_f1": 0.8374
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

### 5. **Transformers**
- Language models for generation like BERT.


---

## 👥 Team Neon

Created as part of the SS25 NLP project module.We aim to make academic research easier and faster using natural Language procesing.

---

## 📌 Notes

- The system is modular and task-specific.
- Easy to extend with new models or embedding techniques.
- Can be integrated into larger QA or chatbot systems.


---

