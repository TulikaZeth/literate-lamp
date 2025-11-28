"""
Notebook Processor - Extract summary, Q&A, and title from Jupyter notebooks
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime


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
    
    def extract_key_points(self, notebook_content: Dict, num_points: int = 5) -> List[str]:
        """
        Extract key points/takeaways from the notebook.
        
        Args:
            notebook_content: Parsed notebook dict
            num_points: Number of key points to extract
            
        Returns:
            List of key points
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing technical content and extracting key insights.
            Extract {num_points} key points/takeaways from this notebook that:
            - Highlight the most important concepts or findings
            - Are concise (1-2 sentences each)
            - Are actionable or memorable
            - Cover different aspects of the notebook
            
            Format as a simple numbered list:
            1. [point]
            2. [point]
            etc."""),
            ("human", """Extract key points from this notebook:

Title: {title}

Content:
{content}

Provide exactly {num_points} key points.""")
        ])
        
        content = notebook_content['all_text'][:4000]
        chain = prompt | self.llm
        response = chain.invoke({
            "title": self.extract_title(notebook_content),
            "content": content,
            "num_points": num_points
        })
        
        # Parse numbered list
        points = []
        for line in response.content.strip().split('\n'):
            # Remove numbering and clean up
            match = re.match(r'^\d+\.\s*(.+)$', line.strip())
            if match:
                points.append(match.group(1))
        
        return points[:num_points]
    
    def extract_references(self, notebook_content: Dict) -> List[Dict[str, str]]:
        """
        Extract libraries, tools, and external references from the notebook.
        
        Args:
            notebook_content: Parsed notebook dict
            
        Returns:
            List of dicts with 'name', 'type', and 'description'
        """
        # Extract imports from code cells
        imports = set()
        for code in notebook_content['code_cells']:
            # Find import statements
            import_matches = re.findall(r'^import\s+(\w+)', code, re.MULTILINE)
            from_matches = re.findall(r'^from\s+(\w+)', code, re.MULTILINE)
            imports.update(import_matches)
            imports.update(from_matches)
        
        # Extract URLs from markdown
        urls = []
        for markdown in notebook_content['markdown_cells']:
            url_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown)
            urls.extend(url_matches)
        
        references = []
        
        # Add libraries
        for lib in sorted(imports):
            references.append({
                "name": lib,
                "type": "library",
                "description": f"Python library used in the notebook"
            })
        
        # Add URL references
        for title, url in urls:
            references.append({
                "name": title,
                "type": "external_link",
                "description": url
            })
        
        return references
    
    def generate_structured_table(self, notebook_content: Dict) -> Dict[str, Any]:
        """
        Generate a structured table summarizing notebook sections.
        
        Args:
            notebook_content: Parsed notebook dict
            
        Returns:
            Dict with table structure
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing notebooks and creating structured summaries.
            Analyze the notebook and create a table that breaks down the content into sections.
            
            For each section, identify:
            - Section name/title
            - Purpose (what it does)
            - Key techniques/methods used
            - Output/result (if applicable)
            
            Format as JSON array:
            [
              {
                "section": "Section name",
                "purpose": "What this section does",
                "techniques": "Key methods used",
                "output": "Result or finding"
              }
            ]
            
            Provide only the JSON array, nothing else."""),
            ("human", """Analyze this notebook and create a structured table:

Title: {title}

Content:
{content}

Provide the JSON array.""")
        ])
        
        content = notebook_content['all_text'][:4000]
        chain = prompt | self.llm
        response = chain.invoke({
            "title": self.extract_title(notebook_content),
            "content": content
        })
        
        try:
            # Try to parse JSON from response
            import json
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                table_data = json.loads(json_match.group())
                return {
                    "columns": ["Section", "Purpose", "Techniques", "Output"],
                    "rows": table_data
                }
        except:
            pass
        
        # Fallback: simple structure
        return {
            "columns": ["Section", "Description"],
            "rows": [
                {
                    "section": "Overview",
                    "description": "Analysis of notebook structure"
                }
            ]
        }
    
    def process_notebook(
        self, 
        notebook_path: str, 
        num_questions: int = 5,
        summary_max_length: int = 300,
        include_structure: bool = True
    ) -> Dict:
        """
        Process notebook to extract title, generate summary, Q&A, and structured analysis.
        
        Args:
            notebook_path: Path to .ipynb file
            num_questions: Number of Q&A pairs to generate
            summary_max_length: Maximum summary length in words
            include_structure: Include structured table and references
            
        Returns:
            Dict with title, summary, qna, key_points, references, table, and metadata
        """
        # Parse notebook
        notebook_content = self.parse_notebook(notebook_path)
        
        # Extract/generate components
        title = self.extract_title(notebook_content)
        summary = self.generate_summary(notebook_content, summary_max_length)
        qna = self.generate_qna(notebook_content, num_questions)
        
        result = {
            "title": title,
            "summary": summary,
            "qna": qna,
            "metadata": {
                "filename": notebook_content['filename'],
                "num_markdown_cells": len(notebook_content['markdown_cells']),
                "num_code_cells": len(notebook_content['code_cells']),
                "total_chars": len(notebook_content['all_text']),
                "processed_at": datetime.now().isoformat()
            }
        }
        
        # Add structured analysis if requested
        if include_structure:
            result["key_points"] = self.extract_key_points(notebook_content)
            result["references"] = self.extract_references(notebook_content)
            result["structure_table"] = self.generate_structured_table(notebook_content)
        
        return result
