import os
import glob
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
    data_dir = settings.DATA_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    yield f"Scanning {data_dir}..."
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        yield f"No PDF files found in {data_dir}."
        return

    documents = []
    for pdf_file in pdf_files:
        yield f"\nProcessing {os.path.basename(pdf_file)}..."
        try:
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            yield f"Total pages: {total_pages}"
            
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
                    yield f"  Loaded {i + 1}/{total_pages} pages..."
                
                # Keep the limit for performance
                if i >= 5:
                    yield "  Stopping at 5 pages for quick testing."
                    break
            
            yield f"  Finished loading {len(file_docs)} pages from {os.path.basename(pdf_file)}."
            documents.extend(file_docs)
            
        except Exception as e:
            yield f"Error reading {os.path.basename(pdf_file)}: {e}"

    if not documents:
        yield "No documents loaded."
        return

    yield f"\nSplitting {len(documents)} document pages..."
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(documents)
    yield f"Created {len(all_splits)} text chunks."

    yield f"Loading embedding model {settings.EMBEDDING_MODEL}..."
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield f"Using device: {device}"
    
    model_kwargs = {"device": device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, 
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=settings.MODEL_CACHE_DIR
    )

    yield f"Ingesting into ChromaDB at {settings.DB_DIR}..."
    
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=settings.DB_DIR)
    
    try:
        # ChromaDB automatically persists in newer versions, but we call it just in case if the API varies
        if hasattr(vectordb, 'persist'):
            vectordb.persist()
    except Exception as e:
        yield f"Warning during persist: {e}"
        
    yield "Ingestion complete! DONE"

def ingest_documents():
    """
    Backwards compatibility: Consumes the generator and prints to console.
    """
    for msg in ingest_documents_generator():
        print(msg)

if __name__ == "__main__":
    ingest_documents()
