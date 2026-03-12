from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

import torch

from med_assistant.core.config import settings
from med_assistant.services.llm_service import load_model_and_pipeline
from langchain.prompts import PromptTemplate

class RAGService:
    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self.vectordb = None

    def initialize(self):
        """
        Initializes the Model and RAG chain. 
        This is heavy and should be done on startup.
        """
        # 1. Load LLM
        hf_pipeline = load_model_and_pipeline()
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

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
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

        template = """
You are a concise medical assistant. Use the context to answer the question briefly.
If unknown, say you do not know. 

Context: {context}

Question: {question}

Answer:"""

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
        
        return self.qa_chain(question)
