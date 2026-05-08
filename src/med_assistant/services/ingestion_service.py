import os
import glob
import json
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pypdf
import torch

from med_assistant.core.config import settings

_HEADING_RE = re.compile(r"^(?P<h>[A-Z][A-Z0-9 \-()/:]{6,})$")

def _normalize_pdf_text(text: str) -> str:
    # Light cleanup to reduce noisy chunk boundaries
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _inject_heading_markers(text: str) -> str:
    """
    Heuristic "structure-aware" hinting: detect likely headings and mark them
    so the splitter prefers breaking around section boundaries.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln:
            out.append("")
            continue
        m = _HEADING_RE.match(ln)
        if m:
            out.append(f"\n\n### {m.group('h').title()}\n")
        else:
            out.append(ln)
    return "\n".join(out).strip()

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
        pdf_basename = os.path.basename(pdf_file)
        doc_title = os.path.splitext(pdf_basename)[0]
        yield json.dumps({"step": "processing", "file": pdf_basename, "message": f"Processing {pdf_basename}..."})
        try:
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            yield json.dumps({"step": "processing", "file": pdf_basename, "message": f"Total pages: {total_pages}"})
            
            file_docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text = _inject_heading_markers(_normalize_pdf_text(text))
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_basename,
                            "doc_title": doc_title,
                            # Keep original `page` for backwards compatibility (0-indexed)
                            "page": i,
                            # Newer metadata (1-indexed for display)
                            "page_number": i + 1,
                            "page_start": i + 1,
                            "page_end": i + 1,
                            "total_pages": total_pages,
                        },
                    )
                    file_docs.append(doc)
                
                if (i + 1) % 50 == 0:
                    yield json.dumps({"step": "processing", "file": pdf_basename, "message": f"  Loaded {i + 1}/{total_pages} pages..."})
                
            yield json.dumps({"step": "processing", "file": pdf_basename, "message": f"  Finished loading {len(file_docs)} pages from {pdf_basename}."})
            documents.extend(file_docs)

            
        except Exception as e:
            yield json.dumps({"step": "processing", "file": pdf_basename, "message": f"Error reading {pdf_basename}: {e}", "status": "error"})

    if not documents:
        yield json.dumps({"step": "processing", "message": "No documents loaded.", "status": "error"})
        return

    yield json.dumps({"step": "splitting", "message": f"Splitting {len(documents)} document pages..."})
    # Prefer splitting around paragraph/section-ish boundaries first.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    all_splits = text_splitter.split_documents(documents)

    # Add per-chunk metadata for better citations/debugging
    chunk_counters: dict[str, int] = {}
    for d in all_splits:
        src = str(d.metadata.get("source", "unknown"))
        idx = chunk_counters.get(src, 0)
        chunk_counters[src] = idx + 1
        d.metadata["chunk_index"] = idx
        d.metadata["chunk_id"] = f"{src}::chunk::{idx}"

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
