# backend/app/services/llm_service.py
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from app.core.config import LLM_MODEL_NAME

# Load the tokenizer and model globally for performance
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"
    # Generate a response
    response = generator(prompt, max_length=100, num_return_sequences=1)
    # Extract the generated answer (remove the prompt part if needed)
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    return answer
