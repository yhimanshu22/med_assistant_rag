import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline
from med_assistant.core.config import settings

class GroqSafeWrapper(ChatGroq):
    """
    A wrapper for ChatGroq that strictly enforces n=1 on all generation calls.
    This prevents Ragas or other libraries from overriding 'n' with values > 1,
    which are not supported by some Groq models.
    """
    def _generate(self, *args, **kwargs):
        kwargs["n"] = 1
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        kwargs["n"] = 1
        return await super()._agenerate(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        kwargs["n"] = 1
        return super().generate(*args, **kwargs)
        
    async def agenerate(self, *args, **kwargs):
        kwargs["n"] = 1
        return await super().agenerate(*args, **kwargs)

def get_llm():

    """
    Returns an LLM instance. 
    Tries Groq first if enabled and API key is present.
    Falls back to local HuggingFace model.
    """
    if settings.USE_GROQ and settings.GROQ_API_KEY:
        print(f"Initializing Groq LLM: {settings.GROQ_MODEL_ID}")
        try:
            return GroqSafeWrapper(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL_ID,
                temperature=0.1,
                max_tokens=4096,
                n=1
            )


        except Exception as e:
            print(f"Failed to initialize Groq, falling back to local: {e}")

    # Fallback to Local
    print("Initializing Local HuggingFace LLM...")
    hf_pipeline = load_local_pipeline()
    return HuggingFacePipeline(pipeline=hf_pipeline)

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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            cache_dir=settings.MODEL_CACHE_DIR
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=settings.MODEL_CACHE_DIR)
    
    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False
    )

