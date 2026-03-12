import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
# Fallback to a smaller model for CPU to ensure it runs
GPU_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Chat"
CPU_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model_and_pipeline():
    """
    Loads the model and tokenizer.
    Adapts to CPU or GPU availability.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device}")

    if device == "cuda":
        model_id = GPU_MODEL_ID
        print(f"Loading {model_id} with 4-bit quantization...")
        try:
            from torch import bfloat16
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )
            model_config = transformers.AutoConfig.from_pretrained(
                model_id, trust_remote_code=True, max_new_tokens=1024
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception as e:
            print(f"Failed to load quantized model: {e}")
            raise e
    else:
        # ** CPU Fallback **
        # Llama-3-8B is too big for most CPUs (RAM usage), so we use TinyLlama for testing/local usage
        model_id = CPU_MODEL_ID
        print(f"Loading {model_id} for CPU usage (Quantization disabled)...")
        # No bitsandbytes config for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="cpu", # Explicitly set to CPU
            torch_dtype=torch.float32 # Use standard float32 for CPU compatibility
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        max_length=2048,
        device_map=device, # "auto" or "cpu"
        repetition_penalty=1.1,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    
    return query_pipeline
