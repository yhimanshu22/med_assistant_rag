import os

# Reduce native crashes on Windows CPU when loading transformers models.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from med_assistant.core.config import settings

torch.set_num_threads(1)


def get_llm():

    """
    Returns a local (open-source) LLM instance backed by HuggingFace Transformers.
    """
    print("Initializing Local HuggingFace LLM...")
    hf_pipeline = load_local_pipeline()
    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

    return local_llm

def load_local_pipeline():
    """
    Loads the local model and tokenizer.
    Adapts to CPU or GPU availability.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Local Device detected: {device}")
    
    # ... existing local loading logic ...
    if device == "cuda":
        model_id = settings.GPU_MODEL_ID
        # ... logic as before ...
        try:
            from torch import bfloat16
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )
            model_config = transformers.AutoConfig.from_pretrained(
                model_id, trust_remote_code=True, max_new_tokens=1024, cache_dir=settings.MODEL_CACHE_DIR
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=settings.MODEL_CACHE_DIR
            )
        except Exception as e:
            print(f"Failed to load quantized model: {e}")
            raise e
    else:
        model_id = settings.CPU_MODEL_ID
        # float32 is slower but far more stable than bfloat16 on Windows CPU.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=torch.float32,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
        model.to("cpu")
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=settings.MODEL_CACHE_DIR)
    
    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.25,
        return_full_text=False
    )

