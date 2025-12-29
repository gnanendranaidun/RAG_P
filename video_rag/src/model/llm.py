
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

class LLMInterface:
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", mode="text", device=None):
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        print(f"Initializing LLM in mode='{mode}' on {self.device}...")
        
        if mode == "video_llava":
            # Placeholder for LLaVA-Video if weights were available publicly in easy HF format
            # Using LLaVA 1.5 as valid proxy for code structure
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            except Exception as e:
                print(f"Failed to load LLaVA model: {e}")
                self.mode = "text_fallback"
        
        if self.mode == "text" or self.mode == "text_fallback":
            # Use specific small model for text-only RAG (Context + Question)
            # Using distilgpt2 for speed/memory reliability
            text_model = "google/flan-t5-base" 
            print(f"Loading text model {text_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(text_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            if "t5" in text_model:
                from transformers import AutoModelForSeq2SeqLM
                self.model = AutoModelForSeq2SeqLM.from_pretrained(text_model).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(text_model).to(self.device)

    def generate(self, prompt, context_text="", image_input=None):
        """
        Generates answer.
        """
        full_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}\nAnswer:"
        
        if self.mode == "video_llava" and image_input is not None:
            # Multi-modal generation
            inputs = self.processor(text=prompt, images=image_input, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=200)
            return self.processor.batch_decode(out, skip_special_tokens=True)[0]
            
        else:
            # Text-only RAG generation
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=200)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    llm = LLMInterface(mode="text")
    print(llm.generate("What is the capital of France?", "France is a country in Europe. Paris is its capital."))
