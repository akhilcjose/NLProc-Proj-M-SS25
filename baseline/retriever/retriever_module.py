from collections import Counter
import os
import re
import faiss
faiss.omp_set_num_threads(1)
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.preprocessing import normalize




class Retriever:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        # Initialize the SentenceTransformer model for generating embeddings
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(reranker_model)  # Load the model here
        self.documents = []  # Store the document chunks
        self.embeddings = []  # Store the embeddings for those chunks
        self.index = None  # FAISS index will be built when documents are added

    def chunk_document(self, document, chunk_size=1000, overlap=400):
        """
        Chunk the document into smaller pieces.
        """
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(document)
    def semantic_chunk_document(self, document, chunk_size=1000, overlap=400):
        """
        Splits the document semantically by section headers, then applies character-based chunking
        to large sections to ensure size constraints.
        """
        # Regex pattern for section headers (customize as needed for your documents)
        section_pattern = re.compile(
            r'(?=\n?\d{1,2}(?:\.\d{1,2})*\s+[A-Z][^\n]+|(?<=\n)[A-Z][A-Za-z ]{3,}\.\n)'
        )

        sections = section_pattern.split(document)
        sections = [sec.strip() for sec in sections if sec.strip()]
    
        # Initialize the text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Split large sections further if needed
        all_chunks = []
        for section in sections:
            if len(section) > chunk_size:
                # Further split large sections
                all_chunks.extend(splitter.split_text(section))
            else:
                all_chunks.append(section)
    
        return all_chunks
    def preprocess_preserve_structure(self,text):
        # 1. Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 2. Remove headers/footers (example: lines that repeat on every page)
        lines = text.split('\n')
        # Optionally, detect and remove lines that are repeated across pages
        # For simplicity, let's skip this unless you have a known pattern

        # 3. Remove page numbers (lines that are just digits)
        lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line)]

        # 4. Remove excessive whitespace but keep paragraph and section breaks
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse 3+ newlines to 2

        # 5. Preserve section headers (e.g., "4.2 Quantitative Evaluation", "Learning Objective.")
        # No action needed if you don't remove all-caps or numbered lines

        # 6. Remove references section (optional, if you want to exclude bibliography)
        text = re.split(r'\nReferences\b|\nREFERENCES\b', text)[0]

        # 7. Remove figure/table captions (optional)
        text = re.sub(r'(Figure|Fig\.|Table) \d+.*\n', '', text)

        # 8. Remove emails and URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+', '', text)

        # 9. Remove equations and special characters (optional, if not needed)
        text = re.sub(r'\$.*?\$', '', text)  # Inline LaTeX equations

        # 10. Normalize spaces again
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text
    
    def read_pdf(self, file_path):
        """
        Extracts text from a PDF file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Preprocess the text to remove unwanted elements
        text = self.preprocess_preserve_structure(text)
        return text
    
    def add_documents(self, document=None, pdf_path=None):
        """
        Add documents to the retriever by chunking them, creating embeddings, and building a FAISS index.
        """
        if document:
            #chunks = self.chunk_document(document)
            chunks = self.semantic_chunk_document(document)
        elif pdf_path:
            text = self.read_pdf(pdf_path)
            #chunks = self.chunk_document(text)
            chunks = self.semantic_chunk_document(text)
        else:
            raise ValueError("Provide either `document` or `pdf_path`.")

        # Generate embeddings for the chunks
        embeddings = self.model.encode(chunks)

        # Store the documents and embeddings
        self.documents.extend(chunks)
        self.embeddings.extend(embeddings)

        # Convert embeddings to a numpy array for FAISS
        embedding_matrix = normalize(np.array(self.embeddings).astype('float32'), axis=1)

        # Create and build the FAISS index
        self.index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        self.index.add(embedding_matrix)

    def query(self, query_text, top_k=20, rerank_k=4):
        """
        Query the retriever to find the most relevant document chunks based on the similarity score.
        """
        # Ensure the index is created before querying
        if self.index is None or len(self.documents) == 0:
            raise ValueError("No documents indexed. Please add documents first.")

        # Generate embedding for the query
        query_embedding = self.model.encode([query_text],normalize_embeddings=True).astype("float32")

         # Perform the search
        D, I = self.index.search(query_embedding, top_k)

        initial_hits = [{"text": self.documents[i], "score": float(D[0][idx])} for idx, i in enumerate(I[0])]
        #print("Initial hits:", initial_hits)

         # Re-rank with cross-encoder
        pairs = [(query_text, chunk['text']) for chunk in initial_hits]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by re-ranking scores
        reranked = sorted(zip(initial_hits, rerank_scores), key=lambda x: x[1], reverse=True)

        return [{"text": chunk['text'], "score": float(score)} for chunk, score in reranked[:rerank_k]]

        # Return the top k document chunks and their similarity scores
        #return [{"text": self.documents[i], "score": float(D[0][idx])} for idx, i in enumerate(I[0])]

    def load(self):
        """
        Load the retriever from a saved state. Placeholder method for loading.
        """
        pass  # Implement loading logic if needed