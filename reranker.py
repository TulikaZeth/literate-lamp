"""
Reranker Module
Implements cross-encoder reranking for improved retrieval quality.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """Rerank retrieved documents using cross-encoder models."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: Optional[int] = None
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model to use for reranking
            top_k: Number of top documents to return after reranking (None = return all)
        """
        self.model_name = model_name
        self.top_k = top_k
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✓ Reranker model loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        return_scores: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User's query
            documents: List of documents to rerank
            return_scores: Whether to return relevance scores
            
        Returns:
            Reranked documents (optionally with scores)
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort documents by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k filter if specified
        if self.top_k:
            doc_score_pairs = doc_score_pairs[:self.top_k]
        
        # Add reranking metadata to documents
        for doc, score in doc_score_pairs:
            doc.metadata['rerank_score'] = float(score)
        
        if return_scores:
            return doc_score_pairs
        else:
            return [doc for doc, _ in doc_score_pairs]
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5
    ) -> List[Document]:
        """
        Rerank and filter documents by relevance threshold.
        
        Args:
            query: User's query
            documents: List of documents to rerank
            threshold: Minimum relevance score (documents below this are filtered out)
            
        Returns:
            Filtered and reranked documents
        """
        doc_score_pairs = self.rerank(query, documents, return_scores=True)
        
        # Filter by threshold
        filtered = [(doc, score) for doc, score in doc_score_pairs if score >= threshold]
        
        return [doc for doc, _ in filtered]


class CohereLLMReranker:
    """Rerank using Cohere's reranking API (more powerful but requires API key)."""
    
    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Cohere reranking model
        """
        try:
            import cohere
            self.client = cohere.Client(api_key)
            self.model = model
            print(f"✓ Cohere reranker initialized with {model}")
        except ImportError:
            raise ImportError("Cohere package not installed. Install with: pip install cohere")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents using Cohere API.
        
        Args:
            query: User's query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        # Prepare documents for Cohere
        texts = [doc.page_content for doc in documents]
        
        # Call Cohere rerank API
        results = self.client.rerank(
            query=query,
            documents=texts,
            top_n=top_k or len(documents),
            model=self.model
        )
        
        # Map results back to original documents
        reranked_docs = []
        for result in results.results:
            doc = documents[result.index]
            doc.metadata['rerank_score'] = result.relevance_score
            reranked_docs.append(doc)
        
        return reranked_docs
