"""
Notebook Processor - Extract summary, Q&A, and title from Jupyter notebooks
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class NotebookProcessor:
    """Process Jupyter notebooks to extract title, summary, and generate Q&A."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize notebook processor with LLM.
        
        Args:
            model_name: Google Gemini model to use
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
    
    def parse_notebook(self, notebook_path: str) -> Dict:
        """
        Parse Jupyter notebook and extract content.
        
        Args:
            notebook_path: Path to .ipynb file
            
        Returns:
            Dict with cells, metadata, and raw content
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            
            cells = notebook_data.get('cells', [])
            metadata = notebook_data.get('metadata', {})
            
            # Extract markdown and code cells
            markdown_cells = []
            code_cells = []
            all_text = []
            
            for cell in cells:
                cell_type = cell.get('cell_type', '')
                source = cell.get('source', [])
                
                # Handle source as list or string
                if isinstance(source, list):
                    content = ''.join(source)
                else:
                    content = source
                
                if cell_type == 'markdown':
                    markdown_cells.append(content)
                    all_text.append(content)
                elif cell_type == 'code':
                    code_cells.append(content)
                    all_text.append(f"```python\n{content}\n```")
            
            return {
                'cells': cells,
                'metadata': metadata,
                'markdown_cells': markdown_cells,
                'code_cells': code_cells,
                'all_text': '\n\n'.join(all_text),
                'filename': os.path.basename(notebook_path)
            }
        
        except Exception as e:
            raise ValueError(f"Error parsing notebook: {str(e)}")
    
    def extract_title(self, notebook_content: Dict) -> str:
        """
        Extract or generate title from notebook.
        
        Priority:
        1. First markdown heading (# Title)
        2. Notebook metadata title
        3. Filename
        4. LLM-generated title
        
        Args:
            notebook_content: Parsed notebook dict
            
        Returns:
            Notebook title
        """
        # Try to find first markdown heading
        for markdown in notebook_content['markdown_cells']:
            match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Try metadata
        metadata = notebook_content.get('metadata', {})
        if 'title' in metadata:
            return metadata['title']
        
        # Fallback to filename without extension
        filename = notebook_content.get('filename', 'Untitled')
        return Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
    
    def generate_summary(self, notebook_content: Dict, max_length: int = 300) -> str:
        """
        Generate a concise summary of the notebook using LLM.
        
        Args:
            notebook_content: Parsed notebook dict
            max_length: Maximum summary length in words
            
        Returns:
            Generated summary
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing Jupyter notebooks and creating concise summaries.
            Create a clear, informative summary that captures:
            - The main topic/purpose of the notebook
            - Key concepts and techniques used
            - Main findings or results (if applicable)
            
            Keep the summary under {max_words} words and make it useful for someone deciding whether to read the full notebook."""),
            ("human", """Analyze this Jupyter notebook and provide a concise summary:

Title: {title}
Filename: {filename}

Content:
{content}

Provide only the summary, no additional commentary.""")
        ])
        
        # Truncate content if too long (keep first 4000 chars to avoid token limits)
        content = notebook_content['all_text'][:4000]
        
        chain = prompt | self.llm
        response = chain.invoke({
            "title": self.extract_title(notebook_content),
            "filename": notebook_content['filename'],
            "content": content,
            "max_words": max_length
        })
        
        return response.content.strip()
    
    def generate_qna(self, notebook_content: Dict, num_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs about the notebook using LLM.
        
        Args:
            notebook_content: Parsed notebook dict
            num_questions: Number of Q&A pairs to generate
            
        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing Jupyter notebooks and creating educational Q&A pairs.
            Generate {num_questions} question-answer pairs that:
            - Cover the most important concepts and techniques
            - Are clear and specific
            - Help someone understand the notebook's content
            - Include both conceptual and practical questions
            
            Format each Q&A as:
            Q: [question]
            A: [answer]
            
            Separate each Q&A pair with a blank line."""),
            ("human", """Analyze this Jupyter notebook and generate {num_questions} Q&A pairs:

Title: {title}
Filename: {filename}

Content:
{content}

Provide only the Q&A pairs in the specified format.""")
        ])
        
        # Truncate content if too long
        content = notebook_content['all_text'][:4000]
        
        chain = prompt | self.llm
        response = chain.invoke({
            "title": self.extract_title(notebook_content),
            "filename": notebook_content['filename'],
            "content": content,
            "num_questions": num_questions
        })
        
        # Parse Q&A pairs from response
        qna_text = response.content.strip()
        qna_pairs = []
        
        # Split by Q: markers
        qa_blocks = re.split(r'\n\s*Q:', qna_text)
        
        for block in qa_blocks:
            if not block.strip():
                continue
            
            # Split by A: marker
            parts = re.split(r'\n\s*A:', block, maxsplit=1)
            
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()
                
                # Remove leading "Q:" if present
                question = re.sub(r'^Q:\s*', '', question)
                
                qna_pairs.append({
                    "question": question,
                    "answer": answer
                })
        
        return qna_pairs[:num_questions]  # Ensure we return exactly num_questions
    
    def process_notebook(
        self, 
        notebook_path: str, 
        num_questions: int = 5,
        summary_max_length: int = 300
    ) -> Dict:
        """
        Process notebook to extract title, generate summary and Q&A.
        
        Args:
            notebook_path: Path to .ipynb file
            num_questions: Number of Q&A pairs to generate
            summary_max_length: Maximum summary length in words
            
        Returns:
            Dict with title, summary, qna, and metadata
        """
        # Parse notebook
        notebook_content = self.parse_notebook(notebook_path)
        
        # Extract/generate components
        title = self.extract_title(notebook_content)
        summary = self.generate_summary(notebook_content, summary_max_length)
        qna = self.generate_qna(notebook_content, num_questions)
        
        return {
            "title": title,
            "summary": summary,
            "qna": qna,
            "metadata": {
                "filename": notebook_content['filename'],
                "num_markdown_cells": len(notebook_content['markdown_cells']),
                "num_code_cells": len(notebook_content['code_cells']),
                "total_chars": len(notebook_content['all_text'])
            }
        }
