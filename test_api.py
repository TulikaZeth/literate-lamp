"""
Test script for FastAPI RAG endpoint
"""

import requests
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_upload_and_query():
    """Test uploading a document and asking a question."""
    print("📤 Testing upload + query...")
    
    # Prepare files (adjust path to your test file)
    files = []
    test_file = Path("test_document.pdf")
    if test_file.exists():
        files.append(("files", open(test_file, "rb")))
    
    # Prepare form data
    data = {
        "question": "What is this document about?",
        "use_ocr": "true"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/rag",
        files=files if files else None,
        data=data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Documents in KB: {result['documents_in_kb']}\n")
    else:
        print(f"Error: {response.text}\n")

def test_query_only():
    """Test querying existing documents."""
    print("❓ Testing query only...")
    
    data = {"question": "Summarize the main points"}
    
    response = requests.post(f"{BASE_URL}/api/rag", data=data)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])}\n")
    else:
        print(f"Error: {response.text}\n")

def test_stats():
    """Test stats endpoint."""
    print("📊 Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/api/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    print("🚀 RAG Bot API Test Suite\n")
    print("=" * 50 + "\n")
    
    test_health()
    test_stats()
    # test_upload_and_query()  # Uncomment if you have a test file
    # test_query_only()        # Uncomment after uploading documents
