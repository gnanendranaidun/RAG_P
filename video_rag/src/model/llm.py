
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

class LLMInterface:
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", mode="text", device=None):
        self.mode = mode
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.model_path = model_path
        
        print(f"Initializing LLM in mode='{mode}' on {self.device}...")
        
        if mode == "video_llava":
            # Placeholder for LLaVA-Video if weights were available publicly in easy HF format
            # Using LLaVA 1.5 as valid proxy for code structure
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            except Exception as e:
                print(f"Failed to load LLaVA model: {e}")
                self.mode = "text_fallback"

        elif mode == "video_moondream":
            print("Loading Moondream2 (Small Efficient LVLM)...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "vikhyatk/moondream2", 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
            except Exception as e:
                print(f"Failed to load Moondream2: {e}")
                self.mode = "text_fallback"
        
        if self.mode == "text" or self.mode == "text_fallback":
            # Revert to distilgpt2 as requested (no download) but optimize generation
            text_model = "distilgpt2" 
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
        # Improved prompt for CausalLM
        full_prompt = f"Context:\n{context_text}\n\nBased on the above context, answer the question.\nQuestion: {prompt}\nAnswer:"
        
        if self.mode == "video_llava" and image_input is not None:
            # Multi-modal generation
            prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(text=prompt_text, images=image_input, return_tensors="pt").to(self.device)
            
            # Generate
            out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            return self.processor.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

        elif self.mode == "video_moondream" and image_input is not None:
            # Moondream generation
            # It prefers: model.encode_image(image) -> model.answer(enc, prompt, tokenizer)
            try:
                enc_image = self.model.encode_image(image_input)
                # Combine context and prompt for Moondream
                # Moondream context instruction works best if embedded in the prompt
                md_prompt = f"{context_text}\nQuestion: {prompt}\nAnswer:" 
                answer = self.model.answer(enc_image, md_prompt, self.tokenizer)
                return answer
            except Exception as e:
                return f"Error in Moondream generation: {e}"
            
        else:
            # Text-only RAG generation
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            # Better generation params to avoid repetition
            out = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                do_sample=False # Greedy decoding for determinism
            )
            
            generated_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Post-processing to extract just the answer
            try:
                answer = generated_text.split("Answer:")[-1].strip()
                # If it continues to generate a new Question:, cut it off
                answer = answer.split("Question:")[0].strip()
                return answer
            except:
                return generated_text

if __name__ == "__main__":
    llm = LLMInterface(mode="text")
    print(llm.generate("What is the capital of France?", "France is a country in Europe. Paris is its capital."))
