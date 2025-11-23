# 🚀 Deploy RAG Bot to Render using Docker

## Prerequisites
- GitHub account
- Render account (free tier available)
- Docker installed locally (for testing)
- Google Gemini API key

---

## 🎯 Deployment Steps

### **Method 1: Using render.yaml (Blueprint) - RECOMMENDED**

This is the easiest method as Render automatically configures everything.

#### 1. **Push to GitHub**
```powershell
# Commit all files
git add .
git commit -m "Add Render Docker deployment"
git push origin main
```

#### 2. **Create Render Blueprint**
- Go to https://dashboard.render.com
- Click **"New"** → **"Blueprint"**
- Connect your GitHub account if not connected
- Select repository: `TulikaZeth/literate-lamp`
- Click **"Apply"**

Render will automatically:
- ✅ Detect `render.yaml`
- ✅ Build Docker image
- ✅ Deploy the service

#### 3. **Add Environment Variables**
After deployment starts:
- Go to your service dashboard
- Click **"Environment"** tab
- Add the following **Secret File**:
  - Key: `GOOGLE_API_KEY`
  - Value: `AIzaSyBYO7t8NV8lIuA384GB9RFtNMS8NGMR3AU`
- Click **"Save Changes"**

#### 4. **Wait for Deployment**
- First build takes 10-15 minutes
- Watch build logs in the dashboard
- Once deployed, you'll get a URL: `https://ragbot-xxxx.onrender.com`

---

### **Method 2: Manual Deployment**

#### 1. **Create New Web Service**
- Go to https://dashboard.render.com
- Click **"New"** → **"Web Service"**
- Connect your GitHub repository: `TulikaZeth/literate-lamp`

#### 2. **Configure Service**
Fill in the following:

**Basic Settings:**
- **Name**: `ragbot`
- **Region**: Oregon (US West) or Frankfurt (EU)
- **Branch**: `main`
- **Runtime**: Docker
- **Root Directory**: (leave empty)

**Docker Settings:**
- **Dockerfile Path**: `./Dockerfile`
- **Docker Build Context**: `.`

**Instance Type:**
- **Plan**: Free (512MB RAM, shared CPU)

#### 3. **Add Environment Variables**

Click **"Advanced"** → Add these environment variables:

```bash
# Required - Add as SECRET
GOOGLE_API_KEY=AIzaSyBYO7t8NV8lIuA384GB9RFtNMS8NGMR3AU

# Optional (already have defaults in render.yaml)
CHAT_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_K=8
RERANKER_TOP_K=4
USE_RERANKER=true
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
USE_OCR=true
PERSIST_DIR=/app/chroma_db
```

#### 4. **Deploy**
- Click **"Create Web Service"**
- Wait for build to complete (10-15 minutes first time)
- Your app will be live at the provided URL

---

## 📊 What Happens During Deployment

```
GitHub Push
    ↓
Render Detects Changes
    ↓
Clone Repository
    ↓
Build Docker Image (10-15 min)
    ├─ Install system packages (tesseract, poppler)
    ├─ Install Python dependencies
    └─ Copy application code
    ↓
Start Container
    ↓
Health Check (/_stcore/health)
    ↓
✅ LIVE at https://ragbot-xxxx.onrender.com
```

---

## 🎨 Features Enabled

✅ **Auto-deploy**: Push to GitHub = automatic deployment  
✅ **HTTPS**: Free SSL certificate included  
✅ **Custom Domain**: Add your own domain (free)  
✅ **Health Checks**: Auto-restart if app crashes  
✅ **Build Cache**: Faster subsequent builds  
✅ **Environment Variables**: Secure secret management  
✅ **Logs**: Real-time streaming logs  

---

## ⚡ Free Tier Limitations

| Feature | Limit |
|---------|-------|
| **RAM** | 512 MB |
| **CPU** | Shared |
| **Build Time** | 15 minutes timeout |
| **Inactivity Sleep** | After 15 minutes |
| **Cold Start** | ~30 seconds |
| **Bandwidth** | 100GB/month |
| **Build Hours** | 750 hours/month |

### Important Notes:
- **Service sleeps after 15 minutes of inactivity**
- **First request wakes it up (~30 seconds)**
- **Storage is ephemeral (resets on each deploy)**
- **Upgrade to paid plan ($7/mo) for persistent disk**

---

## 🔧 Managing Your Deployment

### **View Logs**
```
Dashboard → Your Service → Logs tab
```
Or use Render CLI:
```bash
render logs -s ragbot --tail
```

### **Manual Deploy**
```
Dashboard → Your Service → Manual Deploy → Deploy latest commit
```

### **Suspend Service**
```
Dashboard → Your Service → Settings → Suspend
```

### **Update Environment Variables**
```
Dashboard → Your Service → Environment → Edit
```

---

## 🐛 Troubleshooting

### **Build Fails**

**Check Docker logs:**
```
Dashboard → Logs → Build Logs
```

**Common issues:**
- Missing dependencies in requirements.txt
- Dockerfile syntax errors
- Out of memory during build

**Solutions:**
- Reduce concurrent pip installs
- Use multi-stage builds
- Upgrade to paid plan (more RAM)

### **App Crashes on Startup**

**Check runtime logs:**
```
Dashboard → Logs → Runtime Logs
```

**Common issues:**
- Missing GOOGLE_API_KEY
- Invalid API key
- Port configuration wrong

**Solutions:**
- Verify environment variables
- Check API key is valid
- Ensure PORT=$PORT in Dockerfile

### **Health Check Fails**

**Symptom:** Service keeps restarting

**Solutions:**
- Check `/_stcore/health` endpoint works locally
- Increase health check timeout in render.yaml
- Check app is binding to 0.0.0.0:$PORT

### **Slow Performance**

**Free tier is limited:**
- 512MB RAM (heavy models may struggle)
- Shared CPU
- Cold starts after 15 min inactivity

**Solutions:**
- Upgrade to paid plan ($7/mo for 1GB RAM)
- Use lighter embedding models
- Implement caching
- Use external cron job to prevent sleep

---

## 📈 Upgrade Options

### **Starter Plan ($7/month)**
- 1 GB RAM
- Persistent disk (10GB)
- No sleep
- Priority support

### **Standard Plan ($25/month)**
- 2 GB RAM
- Persistent disk (20GB)
- Faster CPU
- Custom metrics

---

## 🎯 Keep Service Awake (Free Tier)

Use a cron job or monitoring service to ping every 10 minutes:

**UptimeRobot (Free):**
1. Create account at https://uptimerobot.com
2. Add monitor → HTTP(s)
3. URL: `https://ragbot-xxxx.onrender.com/_stcore/health`
4. Interval: 5 minutes

**Cron-job.org (Free):**
1. Create account at https://cron-job.org
2. Create cronjob
3. URL: `https://ragbot-xxxx.onrender.com/_stcore/health`
4. Interval: Every 10 minutes

---

## 🔒 Security Best Practices

1. **Never commit .env with real keys**
   ```bash
   # Add to .gitignore
   .env
   ```

2. **Use Render's Secret Files for API keys**
   - Dashboard → Environment → Add Secret File

3. **Enable GitHub branch protection**
   - Require pull request reviews
   - Prevent direct pushes to main

4. **Regular security updates**
   ```bash
   # Update dependencies
   pip list --outdated
   ```

5. **Monitor usage & logs**
   - Check for suspicious activity
   - Set up alerts

---

## 📊 Monitoring

### **Built-in Metrics**
- Dashboard → Your Service → Metrics
- CPU usage
- Memory usage
- Request count
- Response time

### **Custom Logging**
Add to your Python code:
```python
import logging
logging.info(f"Query: {question}")
logging.info(f"Response time: {time_taken}s")
```

View in Render logs:
```
Dashboard → Logs → Filter by level
```

---

## 🚀 CI/CD Pipeline

Render automatically deploys on:
- Push to main branch
- Pull request merge
- Manual trigger

**Customize in render.yaml:**
```yaml
services:
  - type: web
    name: ragbot
    autoDeploy: true  # Auto-deploy on push
    branch: main       # Deploy from main branch
```

---

## 📦 Alternative: Docker Compose on Render

If you need multiple services (e.g., Redis cache):

1. **Create docker-compose.yml** (already done!)
2. **Update render.yaml:**
```yaml
services:
  - type: web
    name: ragbot
    runtime: docker
    dockerCommand: docker-compose up
```

---

## ✅ Deployment Checklist

Before deploying, ensure:

- [ ] Code pushed to GitHub
- [ ] Dockerfile tested locally (`docker build -t ragbot .`)
- [ ] Docker compose tested (`docker-compose up`)
- [ ] .env not committed to Git
- [ ] render.yaml configured
- [ ] GOOGLE_API_KEY ready
- [ ] requirements.txt up to date
- [ ] Health check endpoint works

---

## 🎉 Post-Deployment

Once deployed:

1. **Test your app**: Upload a PDF and ask questions
2. **Check logs**: Verify no errors
3. **Monitor performance**: Watch memory/CPU
4. **Share URL**: Send to users
5. **Set up monitoring**: UptimeRobot to prevent sleep

---

## 🆘 Get Help

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Render Status**: https://status.render.com
- **Support**: Dashboard → Help → Contact Support

---

**Your RAG Bot is now deployed to Render with Docker!** 🎊

Access at: `https://ragbot-xxxx.onrender.com`
