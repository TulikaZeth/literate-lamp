# Docker deployment script for Windows PowerShell

Write-Host "🐳 RAG Bot - Docker Deployment" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed!" -ForegroundColor Red
    Write-Host "Install from: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is running
docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop" -ForegroundColor Yellow
    exit 1
}

# Check for .env file
if (!(Test-Path .env)) {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    Write-Host "Creating .env template..." -ForegroundColor Yellow
    @"
GOOGLE_API_KEY=your_api_key_here
"@ | Out-File -FilePath .env -Encoding utf8
    Write-Host "✅ Created .env file - Please add your GOOGLE_API_KEY" -ForegroundColor Green
    exit 1
}

# Load environment variables
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}

# Check if API key is set
if ($env:GOOGLE_API_KEY -eq "your_api_key_here" -or [string]::IsNullOrWhiteSpace($env:GOOGLE_API_KEY)) {
    Write-Host "❌ Please set your GOOGLE_API_KEY in .env file!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker is ready" -ForegroundColor Green
Write-Host "✅ Environment configured`n" -ForegroundColor Green

# Build and run
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!`n" -ForegroundColor Green
    
    Write-Host "🚀 Starting RAG Bot..." -ForegroundColor Yellow
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ RAG Bot is running!" -ForegroundColor Green
        Write-Host "`n📱 Access your app at:" -ForegroundColor Cyan
        Write-Host "   http://localhost:8501`n" -ForegroundColor White
        
        Write-Host "📊 Useful commands:" -ForegroundColor Cyan
        Write-Host "   View logs:     docker-compose logs -f" -ForegroundColor Gray
        Write-Host "   Stop app:      docker-compose stop" -ForegroundColor Gray
        Write-Host "   Restart app:   docker-compose restart" -ForegroundColor Gray
        Write-Host "   Remove app:    docker-compose down`n" -ForegroundColor Gray
    } else {
        Write-Host "`n❌ Failed to start container" -ForegroundColor Red
        Write-Host "Check logs: docker-compose logs" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n❌ Build failed!" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
}
