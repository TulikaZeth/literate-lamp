"""
Test script for notebook analysis API endpoints
"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_notebook_analysis(notebook_path: str):
    """Test notebook analysis endpoint."""
    print(f"📓 Testing notebook analysis: {notebook_path}")
    
    with open(notebook_path, 'rb') as f:
        files = {'notebook': (notebook_path, f, 'application/json')}
        response = requests.post(f"{BASE_URL}/api/notebook/analyze", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📌 Title: {result['title']}")
        print(f"\n📝 Summary:\n{result['summary']}")
        print(f"\n❓ Q&A Pairs ({len(result['qna'])} generated):")
        for i, qa in enumerate(result['qna'], 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
        print(f"\n📊 Metadata:")
        print(json.dumps(result['metadata'], indent=2))
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "="*80 + "\n")


def test_custom_qna(notebook_path: str, num_questions: int = 3, summary_length: int = 150):
    """Test custom Q&A endpoint."""
    print(f"📓 Testing custom Q&A: {notebook_path}")
    print(f"   Questions: {num_questions}, Summary length: {summary_length} words")
    
    with open(notebook_path, 'rb') as f:
        files = {'notebook': (notebook_path, f, 'application/json')}
        data = {
            'num_questions': num_questions,
            'summary_length': summary_length
        }
        response = requests.post(f"{BASE_URL}/api/notebook/custom-qna", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📌 Title: {result['title']}")
        print(f"\n📝 Summary:\n{result['summary']}")
        print(f"\n❓ Q&A Pairs ({len(result['qna'])} generated):")
        for i, qa in enumerate(result['qna'], 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test health
    test_health()
    
    # Test notebook analysis
    # Replace with your actual notebook path
    notebook_path = "example_notebook.ipynb"
    
    print("📋 To test the notebook analysis API:")
    print(f"1. Make sure the server is running: uvicorn main:app --reload")
    print(f"2. Create a test notebook or use an existing one")
    print(f"3. Update notebook_path in this script")
    print(f"4. Run: python test_notebook_api.py\n")
    
    # Uncomment these when you have a notebook to test:
    # test_notebook_analysis(notebook_path)
    # test_custom_qna(notebook_path, num_questions=3, summary_length=150)
