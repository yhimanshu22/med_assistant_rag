import os
import glob
import time
from langchain.document_loaders import PyPDFLoader
# Switch to pypdf directly for granular control if needed, 
# but PyPDFLoader is easier for langchain integration.
# We can wrap it or just use pypdf to read page count first.
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Configuration
DATA_DIR = "data"
DB_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def ingest_documents():
    """
    Ingests PDF documents from the DATA_DIR into ChromaDB.
    """
    print(f"Scanning {DATA_DIR}...")
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}.")
        return

    documents = []
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file}...")
        try:
            # 1. Read PDF with pypdf to show progress
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            print(f"Total pages: {total_pages}")
            
            # Extract text page by page
            file_docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={"source": pdf_file, "page": i}
                    )
                    file_docs.append(doc)
                
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{total_pages} pages...")
                
                # Limit for testing speed
                if i >= 50:
                    print("  Stopping at 50 pages for quick testing.")
                    break
            
            print(f"  Finished loading {len(file_docs)} pages from {pdf_file}.")
            documents.extend(file_docs)
            
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")

    if not documents:
        print("No documents loaded.")
        return

    # 2. Split Data
    print(f"\nSplitting {len(documents)} document pages...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(documents)
    print(f"Created {len(all_splits)} text chunks.")

    # 3. Create Embeddings and Vector Store
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

    print(f"Ingesting into ChromaDB at {DB_DIR}...")
    # Process in batches to avoid OOM on large datasets
    batch_size = 5000 # Chroma handles batching, but we pass all at once usually. 
                      # For very large lists, passing all might be heavy.
                      # Chroma.from_documents handles it reasonably well usually.
    
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=DB_DIR)
    
    # Persistence
    try:
        vectordb.persist()
    except:
        pass
        
    print("Ingestion complete!")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    ingest_documents()
