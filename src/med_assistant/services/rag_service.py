from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

import torch

from med_assistant.core.config import settings
from med_assistant.services.llm_service import load_model_and_pipeline
from med_assistant.services.evaluation_service import EvaluatorService
from langchain.prompts import PromptTemplate

class RAGService:
    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self.vectordb = None
        self.evaluator = None # Initialize evaluator attribute

    def initialize(self):
        """
        Initializes the Model and RAG chain. 
        This is heavy and should be done on startup.
        """
        # 1. Load LLM
        hf_pipeline = load_model_and_pipeline()
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # 1.1 Initialize Evaluator
        self.evaluator = EvaluatorService(hf_pipeline)

        # 2. Load Embeddings
        # Check device for embedding model
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL, 
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=settings.MODEL_CACHE_DIR
        )

        # 3. Load VectorDB
        print(f"Loading ChromaDB from {settings.DB_DIR}...")
        self.vectordb = Chroma(persist_directory=settings.DB_DIR, embedding_function=embeddings)

        # 4. Create QA Chain
        # Increased k to 6 and enabled MMR for better context variety and accuracy
        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
        )

        template = """
You are a highly accurate and professional Medical Assistant. 
Your goal is to provide evidence-based answers using ONLY the provided medical context.

INSTRUCTIONS:
1. Base your answer strictly on the provided context. 
2. If the context does not contain enough information to answer the question, state clearly that you do not have enough specific information from the provided documents.
3. Maintain a professional, clinical, and helpful tone.
4. If there are conflicting details in the context, mention them.
5. Do NOT hallucinate or use outside knowledge that isn't supported by the context.

Context:
{context}

Question: 
{question}

Detailed Evidence-Based Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            verbose=True
        )
        print("RAG Service initialized.")

    def answer_question(self, question: str):
        if not self.qa_chain:
            raise RuntimeError("RAG Service is not initialized.")
        
        raw_result = self.qa_chain(question)
        
        answer = raw_result["result"]
        source_docs = raw_result["source_documents"]
        
        # Format sources
        sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in source_docs
        ]
        
        # Perform real-time evaluation for Clinical Reliability
        # Truncate context to a safe length for LLM evaluation (approx 1500 tokens / 6000 chars)
        context_str = "\n".join([doc.page_content for doc in source_docs])
        if len(context_str) > 6000:
            context_str = context_str[:6000] + "... [context truncated]"
            
        eval_results = self.evaluator.evaluate_response(question, context_str, answer)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": eval_results["confidence_score"],
            "metrics": {
                "faithfulness": eval_results["faithfulness"],
                "relevance": eval_results["relevance"]
            }
        }
