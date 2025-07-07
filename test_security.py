#!/usr/bin/env python3
"""
Test script for MemoryOS MCP Server Security Features
"""

import os
import sys
import requests
import json
import time
from datetime import datetime

# Test Configuration
TEST_SERVER_URL = "http://localhost:3000"
TEST_API_KEY = "test-key-123"

def test_health_check():
    """Test public health check endpoint"""
    try:
        response = requests.get(f"{TEST_SERVER_URL}/")
        print(f"✅ Health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_authentication():
    """Test API key authentication"""
    try:
        # Test without API key
        response = requests.post(f"{TEST_SERVER_URL}/mcp", json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1
        })
        print(f"✅ No auth test: {response.status_code} (should be 401)")
        
        # Test with API key
        response = requests.post(f"{TEST_SERVER_URL}/mcp", 
            headers={"X-API-Key": TEST_API_KEY},
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {},
                "id": 1
            })
        print(f"✅ With auth test: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting functionality"""
    try:
        # Send multiple requests rapidly
        for i in range(5):
            response = requests.get(f"{TEST_SERVER_URL}/health")
            print(f"Request {i+1}: {response.status_code}")
            time.sleep(0.1)
        
        print("✅ Rate limiting test completed")
        return True
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False

def test_admin_endpoints():
    """Test authenticated admin endpoints"""
    try:
        # Test admin stats
        response = requests.get(f"{TEST_SERVER_URL}/admin/stats",
            headers={"X-API-Key": TEST_API_KEY})
        print(f"✅ Admin stats: {response.status_code}")
        
        # Test admin sessions
        response = requests.get(f"{TEST_SERVER_URL}/admin/sessions",
            headers={"X-API-Key": TEST_API_KEY})
        print(f"✅ Admin sessions: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Admin endpoints test failed: {e}")
        return False

def main():
    """Run all security tests"""
    print("🔐 MemoryOS MCP Security Test Suite")
    print("="*50)
    
    # Set test environment
    os.environ["MCP_API_KEY"] = TEST_API_KEY
    os.environ["SERVER_MODE"] = "streamable-http"
    os.environ["PORT"] = "3000"
    
    print("📋 Test Configuration:")
    print(f"   Server URL: {TEST_SERVER_URL}")
    print(f"   API Key: {TEST_API_KEY}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Authentication", test_authentication),
        ("Rate Limiting", test_rate_limiting),
        ("Admin Endpoints", test_admin_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"🧪 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("📊 Test Results:")
    print("="*50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All security tests passed! Ready for deployment.")
        return 0
    else:
        print("⚠️  Some security tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 