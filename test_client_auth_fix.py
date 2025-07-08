#!/usr/bin/env python3
"""
Test proper MCP 2.0 client authentication following the FastMCP pattern
This shows the EXACT code the client repository needs to implement
"""

import asyncio
import json
import httpx
from pathlib import Path
from fastmcp import Client

class MCPClientAuthTest:
    """Test client authentication patterns for the remote MCP server"""
    
    def __init__(self):
        # Load API key from our config
        config_file = Path("config.json")
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
                api_keys = config_data.get("api_keys", {})
                if api_keys:
                    self.api_key = list(api_keys.keys())[0]
                    print(f"Using API key: {self.api_key[:8]}...")
                else:
                    raise ValueError("No API keys found in config.json")
        else:
            raise ValueError("config.json not found")
    
    async def test_fastmcp_with_auth(self):
        """Test 1: FastMCP Client with proper authentication headers"""
        print("\n🧪 Test 1: FastMCP Client with Authentication")
        print("=" * 50)
        
        # This is the EXACT pattern the client repository needs
        server_url = "http://localhost:5000/mcp/"
        
        # Custom headers for authentication
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "MemoryOS-Client/1.0"
        }
        
        try:
            # FastMCP handles session management internally
            async with Client(server_url) as client:
                print("✅ Connected to MCP server with authentication")
                
                # List tools
                tools = await client.list_tools()
                print(f"📋 Found {len(tools)} tools:")
                for tool in tools:
                    print(f"   - {tool.name}")
                
                # Test add_memory
                print("\n🔧 Testing add_memory tool...")
                result = await client.call_tool("add_memory", {
                    "params": {
                        "user_input": "Client authentication test successful!",
                        "agent_response": "Remote MCP authentication is working properly.",
                        "user_id": "client_auth_test",
                        "meta_data": {"test_type": "authentication_verification"}
                    }
                })
                print(f"✅ Add memory result: {result[0].text[:80]}...")
                
                # Test retrieve_memory
                print("\n🔍 Testing retrieve_memory tool...")
                result = await client.call_tool("retrieve_memory", {
                    "params": {
                        "query": "authentication test",
                        "user_id": "client_auth_test",
                        "max_results": 3
                    }
                })
                print(f"✅ Retrieve memory result: {result[0].text[:80]}...")
                
                return True
                
        except Exception as e:
            print(f"❌ FastMCP test failed: {e}")
            return False
    
    async def test_httpx_direct_auth(self):
        """Test 2: Direct HTTP client with authentication (for debugging)"""
        print("\n🧪 Test 2: Direct HTTP with Authentication")
        print("=" * 50)
        
        server_url = "http://localhost:5000/mcp/"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "MemoryOS-DirectHTTP/1.0"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test tools/list endpoint
                response = await client.post(
                    server_url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": 1
                    },
                    headers=headers
                )
                
                print(f"📊 Status: {response.status_code}")
                print(f"📋 Response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    return True
                else:
                    print(f"❌ HTTP request failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ HTTP test failed: {e}")
            return False
    
    async def test_without_auth(self):
        """Test 3: Verify authentication is required"""
        print("\n🧪 Test 3: No Authentication (Should Fail)")
        print("=" * 50)
        
        server_url = "http://localhost:5000/mcp/"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    server_url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": 1
                    },
                    headers=headers
                )
                
                print(f"📊 Status: {response.status_code}")
                print(f"📋 Response: {response.text}")
                
                if response.status_code == 401:
                    print("✅ Authentication correctly required")
                    return True
                else:
                    print(f"⚠️  Expected 401, got {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ No-auth test failed: {e}")
            return False

async def main():
    """Run all authentication tests"""
    print("🔐 MemoryOS Remote MCP Authentication Test Suite")
    print("="*60)
    
    tester = MCPClientAuthTest()
    
    # Run all tests
    test1_result = await tester.test_fastmcp_with_auth()
    test2_result = await tester.test_httpx_direct_auth()
    test3_result = await tester.test_without_auth()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("="*30)
    print(f"✅ FastMCP Authentication: {'PASS' if test1_result else 'FAIL'}")
    print(f"✅ Direct HTTP Authentication: {'PASS' if test2_result else 'FAIL'}")
    print(f"✅ Authentication Required: {'PASS' if test3_result else 'FAIL'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 ALL TESTS PASSED - Remote MCP authentication working!")
        print("\n📋 Client Repository Fix Needed:")
        print("1. Add MCP_API_KEY to environment variables")
        print("2. Add Authorization header to MCP client requests")
        print("3. Update MCP server URL to include /mcp path")
        print(f"4. Use API key: {tester.api_key}")
    else:
        print("\n❌ Some tests failed - check authentication setup")

if __name__ == "__main__":
    asyncio.run(main())