"""
Enhanced RAG Bot Streamlit Interface
Multi-modal document Q&A with OCR, reranking, and detailed citations.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from pdf_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Bot - Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False


def initialize_system():
    """Initialize vector store and RAG engine."""
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStoreManager(
            persist_directory=os.getenv("PERSIST_DIR", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            
        )

    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine( 
            vector_store=st.session_state.vector_store,
            model_name=os.getenv("CHAT_MODEL", "gemini-1.5-pro"),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "8")),
            use_reranker=os.getenv("USE_RERANKER", "true").lower() == "true",
            reranker_top_k=int(os.getenv("RERANKER_TOP_K", "4"))
        )


def process_uploaded_files(uploaded_files, use_ocr=True):
    """Process uploaded multi-modal files."""
    processor = DocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        use_ocr=use_ocr
    )
    
    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    # Process documents with progress tracking
    with st.spinner("Processing documents (OCR enabled for images)..."):
        chunks = processor.process_multiple_documents(file_paths)
        if chunks:
            st.session_state.vector_store.add_documents(chunks)
    
    # Clean up temp files
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except:
            pass
    
    st.session_state.documents_loaded = True
    return len(chunks)


def main():
    """Main application."""
    st.title("📚 RAG Bot - Multi-Modal Document Intelligence")
    st.markdown("*Upload PDFs, images, or text files - Ask questions with AI-powered retrieval & citations!*")
    st.caption("Powered by Google Gemini 🤖")
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        st.stop()
    
    # Initialize system
    initialize_system() 
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader with multi-format support
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'md', 'docx'],
            accept_multiple_files=True,
            help="Upload PDFs, images (OCR enabled), text files, or DOCX documents"
        )
        
        use_ocr = st.checkbox("Enable OCR for images/scanned PDFs", value=True)
        
        if uploaded_files and st.button("🚀 Process Documents", type="primary"):
            num_chunks = process_uploaded_files(uploaded_files, use_ocr)
            if num_chunks > 0:
                st.success(f"✅ Processed {len(uploaded_files)} files into {num_chunks} chunks")
            else:
                st.error("❌ No documents were successfully processed")
        
        st.divider()
        
        # Document stats
        st.subheader("📊 Knowledge Base Stats")
        doc_count = st.session_state.vector_store.get_document_count() if st.session_state.vector_store else 0
        st.metric("Total Chunks", doc_count)
        
        if st.button("Clear Knowledge Base", type="secondary"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_vectorstore()
                st.session_state.chat_history = []
                st.session_state.documents_loaded = False
                st.success("✅ Knowledge base cleared")
                st.rerun()
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        show_sources = st.checkbox("Show source documents", value=True)
        show_relevance = st.checkbox("Show relevance scores", value=True)
        st.caption("📊 Reranking: Enabled" if st.session_state.rag_engine and st.session_state.rag_engine.use_reranker else "📊 Reranking: Disabled")
    
    # Main chat interface
    if not st.session_state.documents_loaded and doc_count == 0:
        st.info("👈 Please upload PDF documents to get started!")
    else:
        st.success(f"✅ {doc_count} document chunks loaded and ready!")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message and show_sources:
                    with st.expander(f"📄 View {len(message['sources'])} Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            relevance_badge = f" `{source['relevance']:.3f}`" if source.get('relevance') else ""
                            st.markdown(f"**[{i}] {source['name']}**{relevance_badge}")
                            st.caption(f"Type: {source['type']} | Chunk: {source['chunk']}")
                            st.text(f"📝 {source['preview']}")
                            st.divider()        # Chat input
        if question := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            with st.chat_message("user"):
                st.markdown(question)
            
            # Get answer from RAG engine
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_engine.query(question)
                    
                    st.markdown(result["answer"])
                    
                    # Prepare sources info with enhanced citations
                    sources_info = []
                    if show_sources and result["source_documents"]:
                        with st.expander(f"📄 View {len(result['source_documents'])} Sources"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                source_name = doc.metadata.get('source', 'Unknown')
                                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                                doc_type = doc.metadata.get('doc_type', 'unknown')
                                rerank_score = doc.metadata.get('rerank_score', None)
                                preview = doc.page_content[:250].replace('\n', ' ').strip() + "..."
                                
                                # Display with relevance score if available
                                relevance_badge = f" `Relevance: {rerank_score:.3f}`" if rerank_score and show_relevance else ""
                                st.markdown(f"**[{i}] {source_name}**{relevance_badge}")
                                st.caption(f"Type: {doc_type} | Chunk: {chunk_id}")
                                st.text(f"📝 {preview}")
                                st.divider()
                                
                                sources_info.append({
                                    "name": source_name,
                                    "chunk": chunk_id,
                                    "type": doc_type,
                                    "preview": preview,
                                    "relevance": rerank_score
                                })
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources_info
                    })
        
        # Clear chat button
        if st.session_state.chat_history and st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
