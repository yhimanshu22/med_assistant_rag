import os
import glob
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pypdf
import torch

from med_assistant.core.config import settings

def ingest_documents_generator():
    """
    Ingests PDF documents from the DATA_DIR into ChromaDB, yielding progress updates.
    """
    start_time = time.time()
    data_dir = settings.DATA_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    yield json.dumps({"step": "scanning", "message": f"Scanning {data_dir}..."})
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        yield json.dumps({"step": "scanning", "message": f"No PDF files found in {data_dir}.", "status": "warning"})
        return

    documents = []
    for pdf_file in pdf_files:
        yield json.dumps({"step": "processing", "file": os.path.basename(pdf_file), "message": f"Processing {os.path.basename(pdf_file)}..."})
        try:
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            yield json.dumps({"step": "processing", "file": os.path.basename(pdf_file), "message": f"Total pages: {total_pages}"})
            
            file_docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={"source": os.path.basename(pdf_file), "page": i}
                    )
                    file_docs.append(doc)
                
                if (i + 1) % 50 == 0:
                    yield json.dumps({"step": "processing", "file": os.path.basename(pdf_file), "message": f"  Loaded {i + 1}/{total_pages} pages..."})
                
            yield json.dumps({"step": "processing", "file": os.path.basename(pdf_file), "message": f"  Finished loading {len(file_docs)} pages from {os.path.basename(pdf_file)}."})
            documents.extend(file_docs)

            
        except Exception as e:
            yield json.dumps({"step": "processing", "file": os.path.basename(pdf_file), "message": f"Error reading {os.path.basename(pdf_file)}: {e}", "status": "error"})

    if not documents:
        yield json.dumps({"step": "processing", "message": "No documents loaded.", "status": "error"})
        return

    yield json.dumps({"step": "splitting", "message": f"Splitting {len(documents)} document pages..."})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(documents)
    yield json.dumps({"step": "splitting", "message": f"Created {len(all_splits)} text chunks."})

    yield json.dumps({"step": "embedding", "message": f"Loading embedding model {settings.EMBEDDING_MODEL}..."})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield json.dumps({"step": "embedding", "message": f"Using device: {device}"})
    
    model_kwargs = {"device": device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, 
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=settings.MODEL_CACHE_DIR
    )

    yield json.dumps({"step": "ingesting", "message": f"Ingesting into ChromaDB at {settings.DB_DIR}..."})
    
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=settings.DB_DIR)
    
    try:
        if hasattr(vectordb, 'persist'):
            vectordb.persist()
    except Exception as e:
        yield json.dumps({"step": "ingesting", "message": f"Warning during persist: {e}", "status": "warning"})
        
    duration = time.time() - start_time
    total_time_str = f"{duration:.2f}s"
    yield json.dumps({"step": "complete", "message": "Ingestion complete!", "total_time": total_time_str})

def ingest_documents():
    """
    Backwards compatibility: Consumes the generator and prints to console.
    """
    for msg in ingest_documents_generator():
        print(msg)

if __name__ == "__main__":
    ingest_documents()
