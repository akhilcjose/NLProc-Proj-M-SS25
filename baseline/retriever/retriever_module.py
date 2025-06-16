import os
import faiss
faiss.omp_set_num_threads(1)
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.preprocessing import normalize




class Retriever:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        # Initialize the SentenceTransformer model for generating embeddings
        self.model = SentenceTransformer(model_name)  # Load the model here
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
        return text
    
    def add_documents(self, document=None, pdf_path=None):
        """
        Add documents to the retriever by chunking them, creating embeddings, and building a FAISS index.
        """
        if document:
            chunks = self.chunk_document(document)
        elif pdf_path:
            text = self.read_pdf(pdf_path)
            chunks = self.chunk_document(text)
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

    def query(self, query_text, top_k=3):
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

        # Return the top k document chunks and their similarity scores
        return [{"text": self.documents[i], "score": float(D[0][idx])} for idx, i in enumerate(I[0])]

    def load(self):
        """
        Load the retriever from a saved state. Placeholder method for loading.
        """
        pass  # Implement loading logic if needed
