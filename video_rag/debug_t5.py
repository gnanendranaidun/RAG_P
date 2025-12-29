
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading T5...")
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
