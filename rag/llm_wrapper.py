from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config import LLAMA_MODEL_NAME


class Llama32Local:
    def __init__(self, model_name: str = LLAMA_MODEL_NAME):
        print(f"Loading Llama model: {model_name}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            dtype=torch.float16,
        )

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,               # 🔑 REQUIRED
                temperature=temperature,      # 🔑 NOW USED
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 🔥 CRITICAL FIX: slice off the prompt tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return decoded.strip()
