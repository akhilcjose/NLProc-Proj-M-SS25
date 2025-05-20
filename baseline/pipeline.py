import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from retriever.retriever_module import Retriever
from generator.generator import Generator

def load_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    print("📄 RAG Pipeline: Text File + Question Answering")
    file_path = "/Users/akhiljose/Projects/NLProc_Master_Project/NLProc-Proj-M-SS25/baseline/winnie_the_pooh.txt"

    # Load document
    document = load_txt_file(file_path)
    if not document:
        return

    # Initialize components
    retriever = Retriever()
    retriever.add_documents([document])

    generator = Generator()

    print("\n✅ Document loaded and indexed. You can now ask questions.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("🔎 Ask a question: ")

        if question.strip().lower() == 'exit':
            print("👋 Exiting the pipeline.")
            break
        try:
            retrieved_chunks = retriever.query(question, top_k=3)
            answer = generator.generate_answer(
                task="qa",
                question=question,
                retrieved_chunks=retrieved_chunks
            )
            print(f"🤖 Answer: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # Avoids shutdown crash
    main()