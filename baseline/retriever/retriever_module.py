import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



class Retriever:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        # Initialize the SentenceTransformer model for generating embeddings
        self.model = SentenceTransformer(model_name)  # Load the model here
        self.documents = []  # Store the document chunks
        self.embeddings = []  # Store the embeddings for those chunks
        self.index = None  # FAISS index will be built when documents are added

    def chunk_document(self, document, chunk_size=20, overlap=10):
        """
        Chunk the document into smaller pieces.
        """
        words = document.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def add_documents(self, documents):
        """
        Add documents to the retriever by chunking them, creating embeddings, and building a FAISS index.
        """
        if not documents:
            raise ValueError("No documents provided to add.")

        chunks = []
        for doc in documents:
            chunks += self.chunk_document(doc)

        # Check if model is initialized
        if not hasattr(self, 'model'):
            raise AttributeError("Model is not initialized")

        # Generate embeddings for the chunks
        embeddings = self.model.encode(chunks)

        # Store the documents and embeddings
        self.documents.extend(chunks)
        self.embeddings.extend(embeddings)

        # Convert embeddings to a numpy array for FAISS
        embedding_matrix = np.array(self.embeddings).astype('float32')

        # Create and build the FAISS index
        self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        self.index.add(embedding_matrix)

    def query(self, query_text, top_k=3):
        """
        Query the retriever to find the most relevant document chunks based on the similarity score.
        """
        # Ensure the index is created before querying
        if self.index is None or len(self.documents) == 0:
            raise ValueError("No documents indexed. Please add documents first.")

        # Generate embedding for the query
        query_embedding = self.model.encode([query_text]).astype("float32")

        # Perform the search
        D, I = self.index.search(query_embedding, top_k)

        # Return the top k document chunks and their similarity scores
        return [(self.documents[i], D[0][idx]) for idx, i in enumerate(I[0])]

    def load(self):
        """
        Load the retriever from a saved state. Placeholder method for loading.
        """
        pass  # Implement loading logic if needed

    # Initialize the retriever
from baseline.retriever.retriever_module import Retriever


retriever = Retriever()


# Example documents
documents = [
    "Lakshmi is a software engineer known for her contributions to machine learning.",
    "She works at a top tech company and has been recognized for her achievements."
]

# Add the documents to the retriever
retriever.add_documents(documents)

# Query the retriever
query = "Why Lakshmi is been recognized?"
results = retriever.query(query, top_k=1)

# Display results
for chunk, score in results:
    print(f"Score: {score:.2f} | Chunk: {chunk}")

def read_text_file(filepath):
    """Function to read content from a text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Example Usage
retriever = Retriever()

def read_text_file(filepath):
    """Function to read content from a text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# Read the text file content
file_path = r'C:\Users\lenov\OneDrive\Desktop\Project (4th sem)\NLP_Project_neon\NLProc-Proj-M-SS25\winnie_the_pooh.txt'
document_content = read_text_file(file_path)

# Add the document to the retriever
retriever.add_documents([document_content])

# Query the retriever
query = "can book be available in USA?"
results = retriever.query(query, top_k=1)

# Display results
for chunk, score in results:
    print(f"Score: {score:.2f} | Chunk: {chunk}")
