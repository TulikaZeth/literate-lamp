# 🚀 FastAPI RAG Bot - API Guide

## Overview
Single multimodal endpoint that handles:
- ✅ Document upload (PDF, images, txt, docx)
- ✅ OCR processing
- ✅ Vector storage
- ✅ Q&A with citations

---

## 🎯 Single Endpoint: `/api/rag`

### **Method:** `POST`

### **Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string (form) | Optional* | Question to ask |
| `files` | file[] (multipart) | Optional* | Documents to upload |
| `use_ocr` | boolean (form) | No (default: true) | Enable OCR |
| `clear_kb` | boolean (form) | No (default: false) | Clear KB first |

*At least one of `question` or `files` is required

---

## 📝 Use Cases

### **1. Upload and Ask (Most Common)**
Upload documents and get immediate answers.

**cURL:**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "files=@book.pdf" \
  -F "files=@image.jpg" \
  -F "question=What is the main topic?" \
  -F "use_ocr=true"
```

**Python:**
```python
import requests

files = [
    ("files", open("book.pdf", "rb")),
    ("files", open("image.jpg", "rb"))
]

data = {
    "question": "What is the main topic?",
    "use_ocr": "true"
}

response = requests.post(
    "http://localhost:8000/api/rag",
    files=files,
    data=data
)

result = response.json()
print(result["answer"])
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('files', fileInput.files[0]);
formData.append('question', 'What is this about?');
formData.append('use_ocr', 'true');

const response = await fetch('http://localhost:8000/api/rag', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result.answer);
```

---

### **2. Query Existing Documents**
Ask questions about already uploaded documents.

**cURL:**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "question=Summarize chapter 1"
```

**Python:**
```python
response = requests.post(
    "http://localhost:8000/api/rag",
    data={"question": "Summarize chapter 1"}
)
```

---

### **3. Upload Only (No Question)**
Just upload documents without asking questions.

**cURL:**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "files=@document.pdf"
```

---

### **4. Clear and Reload**
Clear knowledge base and upload new documents.

**cURL:**
```bash
curl -X POST http://localhost:8000/api/rag \
  -F "files=@new_book.pdf" \
  -F "question=What's new?" \
  -F "clear_kb=true"
```

---

## 📊 Response Format

```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "source": "book.pdf",
      "chunk_id": "1",
      "doc_type": "pdf",
      "rerank_score": 0.95,
      "excerpt": "First 300 characters of relevant text..."
    }
  ],
  "metadata": {
    "num_sources": 4,
    "used_reranker": true,
    "model": "gemini-2.5-flash"
  },
  "documents_in_kb": 127
}
```

---

## 🔧 Additional Endpoints

### **Health Check**
```bash
GET /api/health
```
Response:
```json
{
  "status": "healthy",
  "documents": 127,
  "model": "gemini-2.5-flash",
  "reranker_enabled": true
}
```

### **Statistics**
```bash
GET /api/stats
```
Response:
```json
{
  "total_chunks": 127,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chat_model": "gemini-2.5-flash",
  "retrieval_k": 8,
  "reranker_top_k": 4,
  "ocr_enabled": true
}
```

### **Clear Knowledge Base**
```bash
DELETE /api/clear
```
Response:
```json
{
  "status": "success",
  "message": "Knowledge base cleared"
}
```

---

## 🚀 Running Locally

### **Install Dependencies**
```powershell
pip install -r requirements.txt
```

### **Set Environment Variables**
Create `.env` file:
```env
GOOGLE_API_KEY=your-key-here
CHAT_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_K=8
USE_RERANKER=true
RERANKER_TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
USE_OCR=true
PERSIST_DIR=./chroma_db
PORT=8000
```

### **Run Server**
```powershell
# Development (auto-reload)
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

### **Test API**
```powershell
python test_api.py
```

---

## 🐳 Docker Deployment

### **Build Image**
```powershell
docker build -t ragbot-api .
```

### **Run Container**
```powershell
docker run -d \
  -p 8000:8000 \
  -e GOOGLE_API_KEY=your-key \
  -v ${PWD}/chroma_db:/app/chroma_db \
  ragbot-api
```

---

## ☁️ Deploy to Render

### **Using render.yaml (Automatic)**
```powershell
git add .
git commit -m "Add FastAPI deployment"
git push origin main
```

Then in Render dashboard:
1. New → Blueprint
2. Select repository
3. Add `GOOGLE_API_KEY` in Environment
4. Deploy!

Your API will be live at: `https://ragbot-xxxx.onrender.com`

---

## 📡 Interactive API Docs

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

Test all endpoints directly in your browser!

---

## 🎨 Frontend Integration Example

### **React Upload + Query**
```jsx
import { useState } from 'react';

function RAGBot() {
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData();
    formData.append('files', e.target.file.files[0]);
    formData.append('question', e.target.question.value);
    formData.append('use_ocr', 'true');

    const response = await fetch('http://localhost:8000/api/rag', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();
    setAnswer(result.answer);
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="file" name="file" accept=".pdf,.jpg,.png" />
      <input type="text" name="question" placeholder="Ask a question..." />
      <button type="submit" disabled={loading}>
        {loading ? 'Processing...' : 'Ask'}
      </button>
      {answer && <div className="answer">{answer}</div>}
    </form>
  );
}
```

---

## ⚡ Performance Tips

1. **Upload documents once:** Use separate calls for upload and queries
2. **Batch processing:** Upload multiple files in single request
3. **Persistent storage:** Mount `/app/chroma_db` as volume
4. **Caching:** Vector embeddings persist between restarts
5. **OCR toggle:** Disable for text-only PDFs (`use_ocr=false`)

---

## 🔒 Security

- Set `GOOGLE_API_KEY` as environment variable (never commit)
- Use HTTPS in production
- Add rate limiting (Render provides this)
- Validate file types and sizes
- Sanitize user inputs

---

## 🐛 Troubleshooting

### **"No documents in knowledge base"**
- Upload documents first, then query
- Check logs for upload errors

### **"GOOGLE_API_KEY not set"**
- Add to `.env` file locally
- Add to environment variables in Render

### **Slow responses**
- First request after sleep is slow (cold start)
- Upgrade Render plan for better performance
- Reduce `RETRIEVAL_K` for faster search

---

## 📊 API Limits

**Free Tier (Render):**
- 512 MB RAM
- Sleeps after 15 min inactivity
- ~30s cold start
- 100GB bandwidth/month

**Google Gemini API:**
- 2.5-flash: 1M tokens context, 15 RPM free tier
- See: https://ai.google.dev/pricing

---

## 🎉 Quick Test

```bash
# Check health
curl http://localhost:8000/api/health

# Upload and ask (replace with your file)
curl -X POST http://localhost:8000/api/rag \
  -F "files=@sample.pdf" \
  -F "question=What is this about?"
```

---

**Your FastAPI RAG Bot is ready! 🚀**

Access docs at: http://localhost:8000/docs
