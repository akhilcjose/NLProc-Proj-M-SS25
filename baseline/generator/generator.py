from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
torch.set_num_threads(1)
from typing import List, Optional

class Generator:
    def __init__(self, model_name='google/flan-t5-base', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length

    def build_prompt(
        self, 
        task: str,
        question: Optional[str] = None,
        retrieved_chunks: Optional[List[dict]] = None,
        options: Optional[List[str]] = None,
        text_to_classify: Optional[str] = None
    ) -> str:
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks]) if retrieved_chunks else ""

        if task == "qa":
            prompt = (
                "You are an assistant answering questions using only the provided context.\n"
                "If the answer is not in the context, respond with 'I don't know.'\n"
                "Respond in one complete sentence.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )

        elif task == "summarization":
            prompt = (
                "Summarize the following context in one sentence.\n"
                "Do not add external knowledge.\n"
                "If summarization is not possible, respond: 'Summary not possible from provided context.'\n\n"
                f"Context:\n{context}\n\nSummary:"
            )

        elif task == "mcq":
            options_text = "\n".join([f"{chr(97+i)}) {opt}" for i, opt in enumerate(options)])
            prompt = (
                "You are an assistant answering multiple-choice questions using only the provided context.\n"
                "Choose only from the given options.\n"
                "If the answer is not in the context, respond with 'I don't know.'\n"
                "Respond with the letter and option text.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer:"
            )

        elif task == "classification":
            prompt = (
                "You are an assistant that classifies input text as either Offensive or Non-Offensive.\n"
                "Use only the definitions provided in the context.\n"
                "Respond with exactly one label: Offensive or Non-Offensive.\n\n"
                f"Context (definitions):\n{context}\n\nText:\n{text_to_classify}\n\nLabel:"
            )

        else:
            raise ValueError(f"Unknown task: {task}")

        return prompt

    def generate_answer(
        self,
        task: str,
        question: Optional[str] = None,
        retrieved_chunks: Optional[List[dict]] = None,
        options: Optional[List[str]] = None,
        text_to_classify: Optional[str] = None
    ) -> str:
        prompt = self.build_prompt(
            task=task,
            question=question,
            retrieved_chunks=retrieved_chunks,
            options=options,
            text_to_classify=text_to_classify
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True
            )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer