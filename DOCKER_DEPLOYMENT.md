# 🐳 Docker Deployment Guide

## Quick Start (5 minutes)

### Prerequisites
- Docker Desktop installed ([Download](https://docs.docker.com/desktop/install/windows-install/))
- Docker Desktop running
- Git repository cloned

### Option 1: Automated Script (Easiest)

```powershell
# Run the deployment script
.\docker-run.ps1
```

That's it! Your app will be running at **http://localhost:8501**

---

### Option 2: Manual Docker Compose

**1. Create `.env` file** (if not exists)
```powershell
# Copy from .env file or create new:
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

**2. Build and run**
```powershell
docker-compose up -d
```

**3. Access app**
- Open browser: http://localhost:8501

---

### Option 3: Docker Commands (Manual)

**Build image:**
```powershell
docker build -t ragbot:latest .
```

**Run container:**
```powershell
docker run -d `
  --name ragbot `
  -p 8501:8501 `
  -e GOOGLE_API_KEY="your_api_key_here" `
  -v ${PWD}/chroma_db:/app/chroma_db `
  ragbot:latest
```

---

## 📊 Docker Commands Cheat Sheet

### View logs
```powershell
# Follow logs (real-time)
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Control container
```powershell
# Stop
docker-compose stop

# Start
docker-compose start

# Restart
docker-compose restart

# Remove (stop + delete)
docker-compose down
```

### Inspect container
```powershell
# Container status
docker-compose ps

# Resource usage
docker stats ragbot

# Execute commands inside container
docker exec -it ragbot bash
```

### Rebuild after code changes
```powershell
# Rebuild and restart
docker-compose up -d --build

# Force full rebuild (no cache)
docker-compose build --no-cache
docker-compose up -d
```

---

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` or create `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
CHAT_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_K=8
RERANKER_TOP_K=4
USE_RERANKER=true
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
USE_OCR=true
```

### Volumes (Data Persistence)

The `docker-compose.yml` mounts two volumes:
- `./chroma_db` → Vector database storage (persists uploads)
- `./temp_uploads` → Temporary file storage

**Data persists even when container is removed!**

### Port Configuration

Default port: `8501`

To use different port, edit `docker-compose.yml`:
```yaml
ports:
  - "9000:8501"  # Access via localhost:9000
```

---

## 🚀 Deploy to Production

### Cloud Platforms

**1. AWS ECS (Elastic Container Service)**
```bash
# Tag image
docker tag ragbot:latest YOUR_ECR_REPO/ragbot:latest

# Push to ECR
docker push YOUR_ECR_REPO/ragbot:latest

# Deploy using ECS console or CLI
```

**2. Google Cloud Run**
```bash
# Tag for Google Container Registry
docker tag ragbot:latest gcr.io/YOUR_PROJECT/ragbot

# Push
docker push gcr.io/YOUR_PROJECT/ragbot

# Deploy
gcloud run deploy ragbot --image gcr.io/YOUR_PROJECT/ragbot
```

**3. Azure Container Instances**
```bash
# Tag for Azure Container Registry
docker tag ragbot:latest YOUR_REGISTRY.azurecr.io/ragbot

# Push
docker push YOUR_REGISTRY.azurecr.io/ragbot

# Deploy
az container create --resource-group myResourceGroup \
  --name ragbot --image YOUR_REGISTRY.azurecr.io/ragbot
```

**4. DigitalOcean App Platform**
- Connect your GitHub repo
- Select Dockerfile
- Set environment variables
- Deploy!

**5. Render (using Dockerfile)**
- Create new Web Service
- Select "Docker"
- Set environment variables
- Deploy automatically

---

## 🛠️ Troubleshooting

### Port already in use
```powershell
# Check what's using port 8501
netstat -ano | findstr :8501

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or change port in docker-compose.yml
```

### Container exits immediately
```powershell
# Check logs
docker-compose logs

# Common issues:
# - Missing GOOGLE_API_KEY
# - Invalid API key format
# - Missing dependencies
```

### Build fails
```powershell
# Clear Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

### Out of disk space
```powershell
# Clean unused images
docker image prune -a

# Clean everything
docker system prune --volumes
```

### ChromaDB permission errors
```powershell
# Create directories with proper permissions
mkdir -p chroma_db temp_uploads

# Or run container as root (not recommended for production)
```

---

## 📈 Performance Optimization

### Multi-stage Build (Smaller Image)

Create optimized Dockerfile:
```dockerfile
# Builder stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Resource Limits

Add to `docker-compose.yml`:
```yaml
services:
  ragbot:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## 🔒 Security Best Practices

1. **Never commit `.env` with real API keys**
2. **Use Docker secrets for production**
3. **Run as non-root user**
4. **Scan images for vulnerabilities:**
   ```powershell
   docker scan ragbot:latest
   ```
5. **Keep base images updated**
6. **Use specific version tags** (not `latest`)

---

## 📦 Image Distribution

### Push to Docker Hub
```powershell
# Login
docker login

# Tag
docker tag ragbot:latest yourusername/ragbot:v1.0

# Push
docker push yourusername/ragbot:v1.0
```

### Pull and Run
```powershell
# Others can now run your app
docker pull yourusername/ragbot:v1.0
docker run -d -p 8501:8501 \
  -e GOOGLE_API_KEY="their_key" \
  yourusername/ragbot:v1.0
```

---

## ✅ Deployment Checklist

- [ ] Docker Desktop installed and running
- [ ] `.env` file created with valid API key
- [ ] Port 8501 available
- [ ] Sufficient disk space (5GB+)
- [ ] Internet connection for downloading dependencies
- [ ] Code pushed to Git (for version control)

---

**Your RAG Bot is now containerized and ready to deploy anywhere!** 🚀
