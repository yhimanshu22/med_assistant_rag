from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma

import torch

from med_assistant.core.config import settings
from med_assistant.services.llm_service import get_llm
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
        Tries Groq first, then falls back to local.
        """
        # 1. Get LLM (Groq or Local)
        self.llm = get_llm()


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

        # 1.1 Initialize Evaluator using the selected LLM (Groq or Local)
        self.evaluator = EvaluatorService(self.llm, embeddings)

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
6. Use Markdown (headers, bullet points, and bold text) to structure your answer for professional clarity and readability.

Context:
{context}

Question: 
{question}

Detailed Evidence-Based Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            return_source_documents=True,
            verbose=True
        )
        print("RAG Service initialized.")

    def answer_question(self, question: str, chat_history: list = None):
        if not self.qa_chain:
            raise RuntimeError("RAG Service is not initialized.")
        
        # Convert list of dicts to list of tuples for LangChain ConversationalRetrievalChain
        formatted_history = []
        if chat_history:
            for i in range(0, len(chat_history) - 1, 2):
                if i + 1 < len(chat_history):
                    formatted_history.append((chat_history[i]["content"], chat_history[i+1]["content"]))

        try:
            raw_result = self.qa_chain({"question": question, "chat_history": formatted_history})
        except Exception as e:
            print(f"Error in QA Chain: {e}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "metrics": {"faithfulness": 0.0, "relevance": 0.0}
            }
        
        answer = raw_result["answer"]
        source_docs = raw_result.get("source_documents", [])
        # If using a ChatModel like Groq, the result might be an AIMessage object
        if hasattr(answer, 'content'):
            answer = str(answer.content)
        else:
            answer = str(answer)
            
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
