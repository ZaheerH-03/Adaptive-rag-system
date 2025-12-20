from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config import LLAMA_MODEL_NAME


class Llama32Local:
    def __init__(self, model_name: str = LLAMA_MODEL_NAME):
        print(f"Loading Llama model: {model_name}")

        # Quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,               # 4-bit load
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",       # recommended quant
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,   # NEW WAY
            dtype=torch.float16
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
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
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip prompt echo if exists
        return decoded.replace(prompt, "").strip()
