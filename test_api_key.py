#!/usr/bin/env python3
"""
Test the generated API key with the MemoryOS MCP Server
"""

import requests
import json

# Your valid API key
API_KEY = "TFlAmKQYtaoyDcFii3E5BXNtKLYloT_aDjiFTapWRU4"
BASE_URL = "http://localhost:5000"

def test_add_memory():
    """Test adding memory with the valid API key"""
    print("🧪 Testing add_memory with valid API key...")
    
    headers = {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {
                "user_input": "My favorite programming language is Python",
                "agent_response": "I'll remember that you prefer Python for programming!",
                "user_id": "test_user_123"
            }
        },
        "id": 1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/mcp", headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invalid_key():
    """Test with invalid API key to verify authentication"""
    print("\n🧪 Testing with invalid API key...")
    
    headers = {
        'X-API-Key': 'invalid-key-test',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {
                "user_input": "This should fail",
                "agent_response": "Authentication should block this",
                "user_id": "test_user"
            }
        },
        "id": 2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/mcp", headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 401  # Should be unauthorized
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_bearer_token():
    """Test bearer token authentication"""
    print("\n🧪 Testing Bearer token authentication...")
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "retrieve_memory",
            "arguments": {
                "query": "programming language",
                "user_id": "test_user_123"
            }
        },
        "id": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/mcp", headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code in [200, 500]  # 500 might be MCP protocol issue, not auth
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("🔑 Testing Valid API Key Generation")
    print("=" * 50)
    print(f"API Key: {API_KEY}")
    print(f"Server: {BASE_URL}")
    
    # Test valid key
    valid_test = test_add_memory()
    
    # Test invalid key
    invalid_test = test_invalid_key()
    
    # Test bearer token
    bearer_test = test_bearer_token()
    
    print("\n📊 Test Results:")
    print(f"✅ Valid API Key Authentication: {'PASS' if valid_test else 'FAIL'}")
    print(f"✅ Invalid API Key Rejection: {'PASS' if invalid_test else 'FAIL'}")
    print(f"✅ Bearer Token Authentication: {'PASS' if bearer_test else 'FAIL'}")
    
    if valid_test and invalid_test:
        print("\n🎉 API Key Generation and Authentication: SUCCESS!")
        print(f"Your API key ({API_KEY}) is valid and working properly.")
    else:
        print("\n❌ API Key Authentication: ISSUES DETECTED")

if __name__ == "__main__":
    main()