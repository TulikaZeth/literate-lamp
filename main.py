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
from notebook_processor import NotebookProcessor

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
notebook_processor = None


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


def get_notebook_processor():
    """Lazy load notebook processor."""
    global notebook_processor
    if notebook_processor is None:
        print("🔄 Loading notebook processor...")
        notebook_processor = NotebookProcessor(
            model_name=os.getenv("CHAT_MODEL", "gemini-2.5-flash")
        )
    return notebook_processor


@app.on_event("startup")
async def startup_event():
    """Minimal startup - lazy load components on first use."""
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set in environment")
    
    # Auto-migrate from old embeddings if needed
    db_path = "./chroma_db"
    migration_flag = os.path.join(db_path, ".migrated_to_gemini")
    
    if os.path.exists(db_path) and not os.path.exists(migration_flag):
        print("⚠️  Detected old ChromaDB with incompatible embeddings (384-dim)")
        print("🔄 Clearing database to use new Google Gemini embeddings (768-dim)...")
        shutil.rmtree(db_path)
        print("✅ Old database cleared. Please re-upload your documents.")
    
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


@app.post("/api/reset-chroma")
async def reset_chroma():
    """
    Reset ChromaDB to fix dimension mismatch.
    Use this if you get 'expecting embedding with dimension of 384, got 768' error.
    """
    try:
        db_path = os.getenv("PERSIST_DIR", "./chroma_db")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            
            # Reset global instance
            global vector_store
            vector_store = None
            
            # Mark as migrated
            os.makedirs(db_path, exist_ok=True)
            with open(os.path.join(db_path, ".migrated_to_gemini"), "w") as f:
                f.write("Migrated to Google Gemini embeddings (768-dim)")
            
            return {
                "status": "success",
                "message": "ChromaDB reset successfully. Please re-upload your documents."
            }
        else:
            return {
                "status": "info",
                "message": "ChromaDB directory doesn't exist. Nothing to reset."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting ChromaDB: {str(e)}")


class NotebookAnalysisResponse(BaseModel):
    """Response model for notebook analysis."""
    title: str
    summary: str
    qna: List[Dict[str, str]]
    metadata: dict


@app.post("/api/notebook/analyze", response_model=NotebookAnalysisResponse)
async def analyze_notebook(
    notebook: UploadFile = File(..., description="Jupyter notebook file (.ipynb)")
):
    """
    **Analyze Jupyter Notebook - Extract Title, Summary, and Q&A**
    
    Upload a Jupyter notebook (.ipynb) to automatically extract:
    - **Title**: Extracted from first markdown heading or generated
    - **Summary**: AI-generated concise summary of the notebook's content
    - **Q&A**: 5 question-answer pairs covering key concepts
    
    **Use Case:**
    Perfect for creating documentation, study guides, or quick overviews of notebooks.
    
    **Example:**
    ```
    POST /api/notebook/analyze
    notebook: my_analysis.ipynb
    ```
    
    **Response:**
    ```json
    {
      "title": "Data Analysis with Pandas",
      "summary": "This notebook demonstrates...",
      "qna": [
        {
          "question": "What library is used for data manipulation?",
          "answer": "Pandas is used for data manipulation..."
        }
      ],
      "metadata": {
        "filename": "my_analysis.ipynb",
        "num_markdown_cells": 10,
        "num_code_cells": 15
      }
    }
    ```
    """
    try:
        # Validate file extension
        if not notebook.filename.endswith('.ipynb'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a Jupyter notebook (.ipynb)"
            )
        
        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = Path(temp_dir) / notebook.filename
            with open(temp_path, "wb") as f:
                content = await notebook.read()
                f.write(content)
            
            # Process notebook
            processor = get_notebook_processor()
            result = processor.process_notebook(str(temp_path))
            
            return NotebookAnalysisResponse(**result)
        
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing notebook: {str(e)}")


@app.post("/api/notebook/custom-qna")
async def analyze_notebook_custom(
    notebook: UploadFile = File(..., description="Jupyter notebook file (.ipynb)"),
    num_questions: int = Form(default=5, description="Number of Q&A pairs to generate (1-20)", ge=1, le=20),
    summary_length: int = Form(default=300, description="Maximum summary length in words (50-1000)", ge=50, le=1000)
):
    """
    **Analyze Notebook with Custom Parameters**
    
    Similar to `/api/notebook/analyze` but allows customization of:
    - Number of Q&A pairs (1-20)
    - Summary length (50-1000 words)
    
    **Example:**
    ```
    POST /api/notebook/custom-qna
    notebook: my_notebook.ipynb
    num_questions: 10
    summary_length: 500
    ```
    """
    try:
        if not notebook.filename.endswith('.ipynb'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a Jupyter notebook (.ipynb)"
            )
        
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = Path(temp_dir) / notebook.filename
            with open(temp_path, "wb") as f:
                content = await notebook.read()
                f.write(content)
            
            processor = get_notebook_processor()
            result = processor.process_notebook(
                str(temp_path),
                num_questions=num_questions,
                summary_max_length=summary_length
            )
            
            return NotebookAnalysisResponse(**result)
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing notebook: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
