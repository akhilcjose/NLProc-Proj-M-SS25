import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from baseline.retriever.retriever_module import Retriever
from baseline.generator.generator import Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_test_inputs(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading test input JSON: {e}")
        return []

def save_log(log_data, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"Error writing log file: {e}")

def main():
    print("RAG Pipeline: Running on test_inputs.json")

    file_path = "/Users/delnajose/Documents/Semester 4/NLP Project/week 8/NLProc-Proj-M-SS25/baseline/winnie_the_pooh.txt"
    #file_path= "/Users/delnajose/Documents/Semester 4/NLP Project/week 8/NLProc-Proj-M-SS25/baseline/researchPaper.pdf"
    test_input_path = "test_inputs.json"
    log_output_path = "log.json"

    # Load document
    document = load_txt_file(file_path)
    if not document:
        return

    # Load test inputs
    test_data = load_test_inputs(test_input_path)
    if not test_data:
        return

    # Initialize components
    retriever = Retriever()
   
   

    #retriever.add_documents([document])
    retriever.add_documents(document=document)

    generator = Generator()

    log_entries = []

    for idx, test_case in enumerate(test_data):
        question = test_case["question"]
        ground_truth = test_case["answer"]

        print(f"[{idx+1}] Processing: {question}")

        try:
            retrieved_chunks = retriever.query(question, top_k=3)
            generated_promt = generator.build_prompt(
                task="qa",
                question=question,
                retrieved_chunks=retrieved_chunks
            )
            generated_answer = generator.generate_answer(
                task="qa",
                question=question,
                retrieved_chunks=retrieved_chunks
            )

            log_entries.append({
                "question": question,
                "ground_truth_answer": ground_truth,
                "retrieved_chunks": retrieved_chunks,
                "generated_answer": generated_answer
            })

        except Exception as e:
            print(f"Error: {e}")
            log_entries.append({
                "question": question,
                "ground_truth_answer": ground_truth,
                "retrieved_chunks": [],
                "generated_answer": None,
                "error": str(e)
            })

    save_log(log_entries, log_output_path)
    print(f"\nResults saved to {log_output_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()