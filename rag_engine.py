"""
Enhanced RAG Query Engine
Advanced retrieval augmented generation with reranking and detailed citations.
"""

import os
from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from vector_store import VectorStoreManager
from reranker import Reranker

class RAGEngine:
    """Enhanced RAG engine with reranking and detailed citations."""
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        retrieval_k: int = 8,
        use_reranker: bool = True,
        reranker_top_k: int = 4
    ):
        """
        Initialize enhanced RAG engine.
        
        Args:
            vector_store: VectorStoreManager instance
            model_name: Google Gemini model to use
            temperature: Temperature for generation
            retrieval_k: Number of documents to retrieve initially
            use_reranker: Enable reranking for better retrieval
            reranker_top_k: Number of documents after reranking
        """
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        
        # Initialize LLM with Google Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize reranker if enabled
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = Reranker(top_k=reranker_top_k)
                print("✓ Reranker enabled for improved retrieval")
            except Exception as e:
                print(f"⚠ Reranker initialization failed: {e}. Proceeding without reranking.")
                self.use_reranker = False
        
        # Create enhanced prompt with citation instructions
        self.prompt_template = """You are a knowledgeable research assistant that provides accurate, well-cited answers based on provided documents.

Instructions:
1. Answer the question using ONLY information from the provided context
2. Be specific and detailed in your response
3. When referencing information, mention which source it comes from (e.g., "According to [Source 1]...")
4. If multiple sources support a point, mention all relevant sources
5. If the context doesn't contain enough information, clearly state what you know and what you don't
6. If the question is unrelated to the documents, politely explain you can only answer based on the provided materials
7. Maintain objectivity and accuracy - never fabricate information

Context Sources:
{context}

Question: {question}

Detailed Answer with Citations:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Ask a question with enhanced retrieval and reranking.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with 'answer', 'source_documents', and 'metadata'
        """
        # Step 1: Initial retrieval (broader search)
        initial_docs = self.vector_store.similarity_search(
            question,
            k=self.retrieval_k
        )
        
        # Step 2: Rerank if enabled
        if self.use_reranker and self.reranker and initial_docs:
            print(f"  Retrieved {len(initial_docs)} docs, reranking to top {self.reranker_top_k}...")
            relevant_docs = self.reranker.rerank(question, initial_docs)
        else:
            relevant_docs = initial_docs[:self.reranker_top_k]
        
        # Step 3: Prepare context with source labels
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'N/A')
            context_parts.append(
                f"[Source {i}: {source}, Chunk {chunk_id}]\n{doc.page_content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Step 4: Generate answer
        prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        
        # Step 5: Prepare enriched response
        return {
            "answer": response.content,
            "source_documents": relevant_docs,
            "metadata": {
                "num_sources": len(relevant_docs),
                "used_reranker": self.use_reranker and self.reranker is not None,
                "model": self.llm.model
            }
        }
    
    def get_relevant_documents(self, question: str, use_reranking: bool = True) -> List[Document]:
        """
        Get relevant documents for a question without generating an answer.
        
        Args:
            question: User's question
            use_reranking: Whether to apply reranking
            
        Returns:
            List of relevant documents
        """
        docs = self.vector_store.similarity_search(question, k=self.retrieval_k)
        
        if use_reranking and self.use_reranker and self.reranker:
            docs = self.reranker.rerank(question, docs)
        
        return docs[:self.reranker_top_k]
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format the response with detailed answer and citations.
        
        Args:
            result: Result from query method
            
        Returns:
            Formatted string with answer and detailed citations
        """
        output = f"📝 Answer:\n{result['answer']}\n\n"
        
        if result['source_documents']:
            output += "📚 Sources & Citations:\n"
            output += "=" * 60 + "\n"
            
            for i, doc in enumerate(result['source_documents'], 1):
                # Extract metadata
                source = doc.metadata.get('source', 'Unknown')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                doc_type = doc.metadata.get('doc_type', 'unknown')
                rerank_score = doc.metadata.get('rerank_score', None)
                
                # Format source header
                output += f"\n[{i}] {source}\n"
                output += f"    Type: {doc_type} | Chunk: {chunk_id}"
                
                if rerank_score is not None:
                    output += f" | Relevance: {rerank_score:.3f}"
                
                output += "\n"
                
                # Show relevant excerpt
                snippet = doc.page_content[:300].replace('\n', ' ').strip()
                output += f"    Excerpt: \"{snippet}...\"\n"
        
        # Add metadata
        if 'metadata' in result:
            output += "\n" + "=" * 60 + "\n"
            output += f"ℹ️  Query Info: {result['metadata']['num_sources']} sources"
            if result['metadata']['used_reranker']:
                output += " (reranked)"
            output += f" | Model: {result['metadata']['model']}\n"
        
        return output
    
    def format_citation(self, doc: Document, index: int) -> Dict[str, Any]:
        """
        Format a single document as a citation object.
        
        Args:
            doc: Document to format
            index: Citation index number
            
        Returns:
            Dictionary with citation information
        """
        return {
            "index": index,
            "source": doc.metadata.get('source', 'Unknown'),
            "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
            "doc_type": doc.metadata.get('doc_type', 'unknown'),
            "excerpt": doc.page_content[:200].strip(),
            "relevance_score": doc.metadata.get('rerank_score', None),
            "metadata": doc.metadata
        }
