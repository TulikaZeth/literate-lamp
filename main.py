"""
FastAPI RAG Bot - Single Multimodal Endpoint
Handles document upload + Q&A in one endpoint
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pdf_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import RAGEngine

# Load environment
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="RAG Bot API",
    description="Multimodal RAG system with document processing and Q&A",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_store = None
rag_engine = None
document_processor = None


def get_vector_store():
    """Lazy load vector store."""
    global vector_store
    if vector_store is None:
        print("🔄 Loading vector store...")
        vector_store = VectorStoreManager(
            persist_directory=os.getenv("PERSIST_DIR", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        )
    return vector_store


def get_rag_engine():
    """Lazy load RAG engine."""
    global rag_engine
    if rag_engine is None:
        print("🔄 Loading RAG engine...")
        rag_engine = RAGEngine(
            vector_store=get_vector_store(),
            model_name=os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "8")),
            use_reranker=os.getenv("USE_RERANKER", "false").lower() == "true",  # Disable reranker to save memory
            reranker_top_k=int(os.getenv("RERANKER_TOP_K", "4"))
        )
    return rag_engine


def get_document_processor():
    """Lazy load document processor."""
    global document_processor
    if document_processor is None:
        print("🔄 Loading document processor...")
        document_processor = DocumentProcessor(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            use_ocr=os.getenv("USE_OCR", "false").lower() == "true"  # Disable OCR to save memory
        )
    return document_processor


@app.on_event("startup")
async def startup_event():
    """Minimal startup - lazy load components on first use."""
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set in environment")
    
    print("✅ RAG Bot ready (components will load on first use)")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: list
    metadata: dict
    documents_in_kb: int


@app.post("/api/rag")
async def multimodal_rag(
    files: List[UploadFile] = File(default=[], description="Upload: PDF (.pdf), Images (.jpg, .jpeg, .png), Text (.txt, .md), Documents (.docx, .doc)"),
    question: str = Form(default="", description="Question to ask about documents"),
    use_ocr: bool = Form(default=True, description="Enable OCR for images/scanned PDFs"),
    clear_kb: bool = Form(default=False, description="Clear knowledge base before uploading new documents")
):
    """
    **Multimodal RAG Endpoint - All-in-One**
    
    This single endpoint handles:
    1. Document upload (PDF, images, txt, docx) - Optional
    2. Document processing with OCR - Optional
    3. Vector storage
    4. Question answering with citations
    
    **Use Cases:**
    - Upload documents and ask question: `files + question`
    - Ask about existing documents: `question only`
    - Upload documents without question: `files only` (returns confirmation)
    - Clear and reload: `clear_kb=true + files + question`
    
    **Example 1: Upload and Ask**
    ```
    POST /api/rag
    files: book.pdf, image.jpg
    question: "What is the main topic?"
    use_ocr: true
    ```
    
    **Example 2: Ask Only**
    ```
    POST /api/rag
    question: "Summarize chapter 1"
    ```
    
    **Example 3: Just Upload**
    ```
    POST /api/rag
    files: document.pdf
    ```
    """
    
    try:
        # Validate: at least files or question must be provided
        if not files and not question.strip():
            raise HTTPException(
                status_code=400,
                detail="Please provide either 'files' to upload or 'question' to ask, or both."
            )
        
        # Step 1: Clear knowledge base if requested
        if clear_kb:
            get_vector_store().clear_vectorstore()
            print("🗑️ Knowledge base cleared")
        
        # Step 2: Process uploaded files if provided
        if files and len(files) > 0:
            # Validate file types
            ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
            
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Save uploaded files
                file_paths = []
                for uploaded_file in files:
                    if uploaded_file.filename:  # Skip empty files
                        file_ext = Path(uploaded_file.filename).suffix.lower()
                        
                        # Validate file extension
                        if file_ext not in ALLOWED_EXTENSIONS:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                            )
                        
                        file_path = Path(temp_dir) / uploaded_file.filename
                        with open(file_path, "wb") as f:
                            content = await uploaded_file.read()
                            f.write(content)
                        file_paths.append(str(file_path))
                
                # Process documents
                if file_paths:
                    processor = get_document_processor()
                    processor.use_ocr = use_ocr
                    chunks = processor.process_multiple_documents(file_paths)
                    
                    if chunks:
                        get_vector_store().add_documents(chunks)
                        print(f"✅ Processed {len(file_paths)} files into {len(chunks)} chunks")
                    else:
                        raise HTTPException(status_code=400, detail="No content extracted from files")
            
            finally:
                # Cleanup temp files
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Step 3: Answer question if provided
        if question and question.strip():
            # Check if we have documents
            doc_count = get_vector_store().get_document_count()
            if doc_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No documents in knowledge base. Please upload documents first."
                )
            
            # Query RAG engine
            result = get_rag_engine().query(question)
            
            # Format sources
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "source": doc.metadata.get('source', 'Unknown'),
                    "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                    "doc_type": doc.metadata.get('doc_type', 'unknown'),
                    "rerank_score": doc.metadata.get('rerank_score'),
                    "excerpt": doc.page_content[:300].strip()
                })
            
            return QueryResponse(
                answer=result["answer"],
                sources=sources,
                metadata=result["metadata"],
                documents_in_kb=doc_count
            )
        
        else:
            # No question provided - just return upload confirmation
            doc_count = get_vector_store().get_document_count()
            return QueryResponse(
                answer="Documents uploaded successfully. No question provided.",
                sources=[],
                metadata={"upload_only": True},
                documents_in_kb=doc_count
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    doc_count = get_vector_store().get_document_count() if vector_store else 0
    return {
        "status": "healthy",
        "documents": doc_count,
        "model": os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
        "reranker_enabled": os.getenv("USE_RERANKER", "true").lower() == "true"
    }


@app.get("/api/stats")
async def get_stats():
    """Get knowledge base statistics."""
    return {
        "total_chunks": get_vector_store().get_document_count() if vector_store else 0,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
        "chat_model": os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "8")),
        "reranker_top_k": int(os.getenv("RERANKER_TOP_K", "4")),
        "ocr_enabled": os.getenv("USE_OCR", "true").lower() == "true"
    }


@app.delete("/api/clear")
async def clear_knowledge_base():
    """Clear all documents from knowledge base."""
    try:
        get_vector_store().clear_vectorstore()
        return {"status": "success", "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
