from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name='google/flan-t5-base', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length

    def build_prompt(self, question: str, retrieved_chunks: List[dict]) -> str:
        # Combine context from retrieved chunks
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return prompt

    def generate_answer(self, question: str, retrieved_chunks: List[dict]) -> str:
        prompt = self.build_prompt(question, retrieved_chunks)
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