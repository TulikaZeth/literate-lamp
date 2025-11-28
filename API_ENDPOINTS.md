# 📡 API Endpoints Reference

Complete documentation of all available endpoints in the RAG Bot API.

---

## 📚 Document RAG Endpoints

### 1. **POST `/api/rag`** - Main Multimodal Endpoint

**Purpose**: All-in-one endpoint for document upload, processing, and Q&A.

**What it does**:
- Accepts multiple file uploads (PDF, images, text, docx)
- Processes documents with optional OCR
- Stores content in vector database
- Answers questions using RAG (Retrieval-Augmented Generation)
- Returns answers with source citations

**Use Cases**:
- Upload documents and ask questions in one request
- Query existing documents without uploading
- Upload documents for later querying
- Clear and reload knowledge base

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `files` | File[] | No* | [] | Documents to upload (.pdf, .txt, .md, .docx, .jpg, .png) |
| `question` | string | No* | "" | Question to ask about documents |
| `use_ocr` | boolean | No | true | Enable OCR for scanned PDFs/images (local only) |
| `clear_kb` | boolean | No | false | Clear knowledge base before uploading |

*At least one of `files` or `question` must be provided

**Example 1: Upload and Ask**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "files=@research_paper.pdf" \
  -F "question=What are the main findings?"
```

**Example 2: Query Only**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "question=Summarize the methodology"
```

**Example 3: Upload Only**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response**:
```json
{
  "answer": "The main findings indicate that...",
  "sources": [
    {
      "source": "research_paper.pdf",
      "chunk_id": "3",
      "doc_type": "pdf",
      "rerank_score": 0.95,
      "excerpt": "The study found that implementing..."
    }
  ],
  "metadata": {
    "num_sources": 4,
    "model": "gemini-2.5-flash"
  },
  "documents_in_kb": 25
}
```

---

### 2. **GET `/api/health`** - Health Check

**Purpose**: Quick check if the API is running and operational.

**What it does**:
- Confirms API is responsive
- Returns document count
- Shows configuration (model, reranker status)

**Use Cases**:
- Monitoring and uptime checks
- Pre-deployment validation
- Status dashboards

**Example**:
```bash
curl http://localhost:8000/api/health
```

**Response**:
```json
{
  "status": "healthy",
  "documents": 25,
  "model": "gemini-2.5-flash",
  "reranker_enabled": false
}
```

---

### 3. **GET `/api/stats`** - Statistics

**Purpose**: Detailed statistics about the knowledge base and configuration.

**What it does**:
- Shows total document chunks in vector database
- Displays current model configurations
- Returns retrieval parameters

**Use Cases**:
- Debugging configuration issues
- Monitoring knowledge base size
- API documentation

**Example**:
```bash
curl http://localhost:8000/api/stats
```

**Response**:
```json
{
  "total_chunks": 127,
  "embedding_model": "models/text-embedding-004",
  "chat_model": "gemini-2.5-flash",
  "retrieval_k": 8,
  "reranker_top_k": 4,
  "ocr_enabled": false
}
```

---

### 4. **DELETE `/api/clear`** - Clear Knowledge Base

**Purpose**: Remove all documents from the vector database.

**What it does**:
- Deletes all stored document chunks
- Clears vector embeddings
- Preserves database structure

**Use Cases**:
- Reset before uploading new dataset
- Clear sensitive data
- Testing and development

**Example**:
```bash
curl -X DELETE http://localhost:8000/api/clear
```

**Response**:
```json
{
  "status": "success",
  "message": "Knowledge base cleared"
}
```

---

### 5. **POST `/api/reset-chroma`** - Reset ChromaDB

**Purpose**: Fix dimension mismatch errors when switching embedding models.

**What it does**:
- Completely removes ChromaDB directory
- Resets vector store to fresh state
- Marks as migrated to new embeddings

**Use Cases**:
- Fix "expecting embedding with dimension of 384, got 768" error
- Migrate from old embedding model (MiniLM) to new (Gemini)
- Complete database reset

**Example**:
```bash
curl -X POST http://localhost:8000/api/reset-chroma
```

**Response**:
```json
{
  "status": "success",
  "message": "ChromaDB reset successfully. Please re-upload your documents."
}
```

**When to use**: You'll see this error if you have old embeddings:
```
Error: Collection expecting embedding with dimension of 384, got 768
```

---

## 📓 Notebook Analysis Endpoints

### 6. **POST `/api/notebook/analyze`** - Complete Notebook Analysis

**Purpose**: Full analysis with title, summary, Q&A, key points, references, and structure.

**What it does**:
- Extracts title from notebook (first heading or auto-generates)
- Generates AI summary of notebook content
- Creates Q&A pairs from code and markdown
- Extracts key points and bullet summaries
- Lists all tables, figures, and references
- Provides cell-by-cell structure breakdown

**Use Cases**:
- Create comprehensive documentation from notebooks
- Generate study materials with Q&A
- Catalog notebook collections
- Extract structured data for reports

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `notebook` | File | Yes | - | Jupyter notebook file (.ipynb) |
| `num_questions` | int | No | 5 | Number of Q&A pairs to generate (1-20) |
| `summary_length` | int | No | 300 | Max summary length in words (50-1000) |

**Example**:
```bash
curl -X POST http://localhost:8000/api/notebook/analyze \
  -F "notebook=@data_analysis.ipynb" \
  -F "num_questions=10" \
  -F "summary_length=500"
```

**Response**:
```json
{
  "title": "Data Analysis with Pandas",
  "summary": "This notebook demonstrates comprehensive data cleaning and exploratory data analysis using the Pandas library...",
  "qna": [
    {
      "question": "What library is used for data manipulation?",
      "answer": "Pandas is the primary library used for data manipulation and analysis in this notebook."
    },
    {
      "question": "How are missing values handled?",
      "answer": "Missing values are handled using fillna() for imputation and dropna() for removal..."
    }
  ],
  "key_points": [
    "Data loading from CSV using pd.read_csv()",
    "Missing value handling with fillna() and dropna()",
    "Data visualization using matplotlib and seaborn",
    "Statistical analysis with describe() and groupby()"
  ],
  "references": [
    {
      "type": "table",
      "title": "DataFrame Summary Statistics",
      "cell_number": 5,
      "description": "Output of df.describe() showing mean, std, min, max"
    },
    {
      "type": "figure",
      "title": "Correlation Heatmap",
      "cell_number": 12,
      "description": "Seaborn heatmap showing feature correlations"
    }
  ],
  "structure_table": {
    "total_cells": 25,
    "markdown_cells": 10,
    "code_cells": 15,
    "breakdown": [
      {"cell": 1, "type": "markdown", "preview": "# Introduction to Data Analysis"},
      {"cell": 2, "type": "code", "preview": "import pandas as pd\nimport numpy as np"}
    ]
  },
  "metadata": {
    "filename": "data_analysis.ipynb",
    "num_markdown_cells": 10,
    "num_code_cells": 15,
    "processed_at": "2025-11-28T10:30:00"
  }
}
```

---

### 7. **POST `/api/notebook/summary`** - Summary Only

**Purpose**: Quick extraction of title, summary, and key points only.

**What it does**:
- Extracts or generates notebook title
- Creates concise summary
- Lists key takeaways as bullet points
- Skips Q&A and detailed structure (faster response)

**Use Cases**:
- Quick previews of notebooks
- Generate documentation overviews
- Batch process large collections
- Create notebook catalogs

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `notebook` | File | Yes | - | Jupyter notebook file (.ipynb) |
| `summary_length` | int | No | 300 | Max summary length in words (50-1000) |

**Example**:
```bash
curl -X POST http://localhost:8000/api/notebook/summary \
  -F "notebook=@ml_tutorial.ipynb" \
  -F "summary_length=200"
```

**Response**:
```json
{
  "title": "Machine Learning Tutorial",
  "summary": "This notebook provides a comprehensive introduction to machine learning using scikit-learn. It covers data preprocessing, model training, evaluation, and prediction.",
  "key_points": [
    "Introduction to supervised learning concepts",
    "Data preprocessing with StandardScaler",
    "Model training using Random Forest classifier",
    "Performance evaluation with accuracy, precision, recall",
    "Hyperparameter tuning with GridSearchCV"
  ],
  "metadata": {
    "filename": "ml_tutorial.ipynb",
    "num_markdown_cells": 8,
    "num_code_cells": 12,
    "processed_at": "2025-11-28T10:35:00"
  }
}
```

---

### 8. **POST `/api/notebook/qna`** - Q&A Only

**Purpose**: Generate only question-answer pairs from notebook.

**What it does**:
- Extracts notebook title
- Generates targeted Q&A pairs covering key concepts
- Skips summary and structure (faster response)
- Perfect for creating quizzes or study guides

**Use Cases**:
- Create quiz questions for educational content
- Generate study materials and flashcards
- Test comprehension of notebook concepts
- Build FAQ sections from technical notebooks

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `notebook` | File | Yes | - | Jupyter notebook file (.ipynb) |
| `num_questions` | int | No | 5 | Number of Q&A pairs to generate (1-20) |

**Example**:
```bash
curl -X POST http://localhost:8000/api/notebook/qna \
  -F "notebook=@python_basics.ipynb" \
  -F "num_questions=10"
```

**Response**:
```json
{
  "title": "Python Basics Tutorial",
  "qna": [
    {
      "question": "What data types are covered in this notebook?",
      "answer": "The notebook covers strings, integers, floats, lists, tuples, dictionaries, and sets."
    },
    {
      "question": "How do you create a list in Python?",
      "answer": "Lists are created using square brackets: my_list = [1, 2, 3, 4]"
    },
    {
      "question": "What is the difference between a list and a tuple?",
      "answer": "Lists are mutable (can be changed) while tuples are immutable (cannot be changed after creation)."
    }
  ],
  "metadata": {
    "filename": "python_basics.ipynb",
    "num_questions": 10,
    "num_markdown_cells": 15,
    "num_code_cells": 20,
    "processed_at": "2025-11-28T10:40:00"
  }
}
```

---

## 📊 Endpoint Comparison Table

| Endpoint | Purpose | Response Time | Use Case |
|----------|---------|---------------|----------|
| `/api/rag` | Document Q&A | 2-5s | Main RAG functionality |
| `/api/health` | Status check | <100ms | Monitoring |
| `/api/stats` | Configuration | <100ms | Debugging |
| `/api/clear` | Clear DB | <500ms | Reset data |
| `/api/reset-chroma` | Fix DB | 1-2s | Migration/fixes |
| `/api/notebook/analyze` | Full analysis | 10-30s | Complete documentation |
| `/api/notebook/summary` | Quick overview | 3-10s | Fast previews |
| `/api/notebook/qna` | Study guide | 5-15s | Quiz generation |

---

## 🔄 Workflow Examples

### **Workflow 1: Building a Knowledge Base**

```bash
# 1. Clear old data
curl -X DELETE http://localhost:8000/api/clear

# 2. Upload documents
curl -X POST http://localhost:8000/api/rag \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf"

# 3. Check stats
curl http://localhost:8000/api/stats

# 4. Ask questions
curl -X POST http://localhost:8000/api/rag \
  -F "question=What are the key points?"
```

### **Workflow 2: Notebook Documentation Pipeline**

```bash
# 1. Get quick summary
curl -X POST http://localhost:8000/api/notebook/summary \
  -F "notebook=@analysis.ipynb"

# 2. Generate Q&A for study guide
curl -X POST http://localhost:8000/api/notebook/qna \
  -F "notebook=@analysis.ipynb" \
  -F "num_questions=20"

# 3. Get full analysis with references
curl -X POST http://localhost:8000/api/notebook/analyze \
  -F "notebook=@analysis.ipynb" \
  -F "num_questions=10" \
  -F "summary_length=500"
```

---

## 🎯 Best Practices

1. **Use `/api/rag` for all document Q&A** - Single endpoint, multiple capabilities
2. **Use `/api/notebook/summary`** first - Quick preview before full analysis
3. **Use `/api/notebook/qna`** for education - Faster than full analysis
4. **Use `/api/notebook/analyze`** for documentation - Complete structured output
5. **Monitor with `/api/health`** - Set up uptime checks
6. **Check `/api/stats`** for debugging - Verify configuration

---

## 🚀 Interactive Testing

Visit **Swagger UI** at http://localhost:8000/docs to test all endpoints interactively in your browser!

---

**Total Endpoints: 8** (5 Document RAG + 3 Notebook Analysis)
