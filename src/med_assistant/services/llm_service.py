import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline
from typing import List, Optional, Any, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
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

class RobustFallbackLLM(BaseChatModel):
    primary: Any
    secondary: Any

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Simple formatter to convert messages to a string for local LLMs."""
        prompt = ""
        for m in messages:
            role = "Human" if m.type == "human" else "Assistant" if m.type == "ai" else "System"
            prompt += f"{role}: {m.content}\n\n"
        return prompt.strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            # Force n=1 for ChatGroq
            kwargs["n"] = 1
            if hasattr(self.primary, "_generate"):
                return self.primary._generate(messages, stop, run_manager, **kwargs)
            return self.primary.invoke(messages, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "overloaded" in str(e).lower() or "not completed" in str(e).lower():
                print(f"!!! Groq Rate Limit/Error: {e}. Falling back to LOCAL LLM.")
                
                # Format prompt for secondary (usually a BaseLLM like HuggingFacePipeline)
                prompt = self._format_messages(messages)
                res = self.secondary.invoke(prompt, stop=stop)
                
                from langchain_core.outputs import ChatGeneration
                from langchain_core.messages import AIMessage
                
                content = res if isinstance(res, str) else str(res)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            raise e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            # Force n=1 for ChatGroq
            kwargs["n"] = 1
            if hasattr(self.primary, "_agenerate"):
                return await self.primary._agenerate(messages, stop, run_manager, **kwargs)
            return await self.primary.ainvoke(messages, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "overloaded" in str(e).lower() or "not completed" in str(e).lower():
                print(f"!!! Groq Async Rate Limit/Error: {e}. Falling back to LOCAL LLM.")
                
                prompt = self._format_messages(messages)
                res = await self.secondary.ainvoke(prompt, stop=stop)
                
                from langchain_core.outputs import ChatGeneration
                from langchain_core.messages import AIMessage
                
                content = res if isinstance(res, str) else str(res)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            raise e

    @property
    def _llm_type(self) -> str:
        return "robust_fallback_llm"


def get_llm():

    """
    Returns an LLM instance. 
    Tries Groq first if enabled and API key is present.
    Falls back to local HuggingFace model.
    """
    # 1. Initialize Groq
    groq_llm = None
    if settings.USE_GROQ and settings.GROQ_API_KEY:
        print(f"Initializing Groq LLM: {settings.GROQ_MODEL_ID}")
        try:
            groq_llm = GroqSafeWrapper(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL_ID,
                temperature=0.1,
                max_tokens=4096,
                n=1
            )
        except Exception as e:
            print(f"Failed to initialize Groq, using local as primary: {e}")

    # 2. Initialize Local (Either as fallback or primary)
    print("Initializing Local HuggingFace LLM...")
    hf_pipeline = load_local_pipeline()
    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 3. Return Fallback Wrapper or Local Primary
    if groq_llm:
        # Use our custom RobustFallbackLLM that works better with Ragas/LangChain internals
        return RobustFallbackLLM(primary=groq_llm, secondary=local_llm)
    
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

