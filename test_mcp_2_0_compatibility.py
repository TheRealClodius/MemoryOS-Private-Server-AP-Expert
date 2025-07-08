#!/usr/bin/env python3
"""
Test MCP 2.0 Compatibility with nested parameter structure
"""

import asyncio
import httpx
import json
import time

async def test_mcp_2_0_compatibility():
    """Test both MCP 2.0 client format and direct parameter format"""
    
    base_url = "http://localhost:5000"
    api_key = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    
    print("🔬 Testing MCP 2.0 Compatibility")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    
    async with httpx.AsyncClient() as client:
        # 1. Initialize session
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "MCP2.0TestClient", "version": "1.0.0"}
            }
        }
        
        print("1. Initializing MCP session...")
        response = await client.post(f"{base_url}/mcp/", json=init_payload, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Init failed: {response.text}")
            return
        
        # Extract session ID from response
        session_id = response.headers.get('mcp-session-id')
        if session_id:
            headers["MCP-Session-ID"] = session_id
            print(f"   ✅ Session ID: {session_id}")
        
        # Wait for initialization to complete
        await asyncio.sleep(0.5)
        
        # 2. Test MCP 2.0 client format (what the client is sending)
        print("\n2. Testing MCP 2.0 client format...")
        test_payload_mcp2 = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "add_memory",
                "arguments": {
                    "params": {  # This is the MCP 2.0 nested structure
                        "user_input": "I love coding in Python",
                        "agent_response": "Python is a great language for many applications!",
                        "user_id": "mcp_2_0_user"
                    }
                }
            }
        }
        
        response = await client.post(f"{base_url}/mcp/", json=test_payload_mcp2, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ MCP 2.0 format works: {result}")
            except:
                print(f"   Response: {response.text}")
        else:
            print(f"   ❌ MCP 2.0 format failed: {response.text}")
        
        # 3. Test direct parameter format (FastMCP style)
        print("\n3. Testing direct parameter format...")
        test_payload_direct = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "add_memory",
                "arguments": {  # Direct parameters, no "params" wrapper
                    "user_input": "I love JavaScript too",
                    "agent_response": "JavaScript is versatile for web development!",
                    "user_id": "direct_user"
                }
            }
        }
        
        response = await client.post(f"{base_url}/mcp/", json=test_payload_direct, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ Direct format works: {result}")
            except:
                print(f"   Response: {response.text}")
        else:
            print(f"   ❌ Direct format failed: {response.text}")
        
        # 4. Test retrieve_memory with MCP 2.0 format
        print("\n4. Testing retrieve_memory with MCP 2.0 format...")
        retrieve_payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "retrieve_memory",
                "arguments": {
                    "params": {
                        "query": "programming languages",
                        "user_id": "mcp_2_0_user"
                    }
                }
            }
        }
        
        response = await client.post(f"{base_url}/mcp/", json=retrieve_payload, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ Retrieve works: {result}")
            except:
                print(f"   Response: {response.text}")
        else:
            print(f"   ❌ Retrieve failed: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_mcp_2_0_compatibility())