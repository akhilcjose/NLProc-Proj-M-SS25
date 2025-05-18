import os
import faiss
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


class Retriever:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.index = None

    def _chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def _read_file(self, file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.txt' or ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def add_documents(self, file_paths: List[str]):
        for file_path in file_paths:
            text = self._read_file(file_path)
            chunks = self._chunk_text(text)
            self.documents.extend([{'text': chunk, 'source': file_path} for chunk in chunks])
            self.chunks.extend(chunks)

        self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, question: str, top_k=3):
        query_embedding = self.embedder.encode([question])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.documents[i] for i in indices[0]]
        return results

    def save(self, dir_path='retriever_store'):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, 'faiss.index'))
        with open(os.path.join(dir_path, 'documents.pkl'), 'wb') as f:
            pickle.dump((self.documents, self.chunks), f)

    def load(self, dir_path='retriever_store'):
        self.index = faiss.read_index(os.path.join(dir_path, 'faiss.index'))
        with open(os.path.join(dir_path, 'documents.pkl'), 'rb') as f:
            self.documents, self.chunks = pickle.load(f)
        self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)

retriever = Retriever()
retriever.add_documents([r'C:\Users\lenov\OneDrive\Desktop\Project (4th sem)\NLP_Project_neon\NLProc-Proj-M-SS25\winnie_the_pooh.txt'])

retriever.save()

retriever.load()

results = retriever.query("What is the main topic of the report?")
for r in results:
    print(r['text'][:200], '\n---')