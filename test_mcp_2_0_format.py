#!/usr/bin/env python3
"""
Test MCP 2.0 JSON-RPC format with exact client structure
"""

import asyncio
import httpx
import json

async def test_mcp_2_0_format():
    """Test the exact MCP 2.0 format the client is sending"""
    
    base_url = "http://localhost:5000"
    api_key = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    
    print("🧪 Testing MCP 2.0 JSON-RPC Format")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    
    async with httpx.AsyncClient() as client:
        # Initialize session
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "TestClient", "version": "1.0.0"}
            }
        }
        
        print("1. Initializing MCP session...")
        response = await client.post(f"{base_url}/mcp/", json=init_payload, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            session_id = response.headers.get('mcp-session-id')
            if session_id:
                headers["MCP-Session-ID"] = session_id
                print(f"   ✅ Session ID: {session_id}")
            
            # Test the exact format client is sending
            print("\n2. Testing add_memory with client format...")
            tool_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "add_memory",
                    "arguments": {
                        "params": {  # This is what client sends
                            "user_input": "I love pizza",
                            "agent_response": "That's great! Pizza is delicious.",
                            "user_id": "test_mcp_2_0_user"
                        }
                    }
                }
            }
            
            response = await client.post(f"{base_url}/mcp/", json=tool_payload, headers=headers)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result}")
            else:
                print(f"   ❌ Error: {response.text}")
        
        print("\n3. Testing direct parameter format...")
        tool_payload_direct = {
            "jsonrpc": "2.0", 
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "add_memory",
                "arguments": {  # Direct parameters (no "params" wrapper)
                    "user_input": "I love pasta",
                    "agent_response": "Pasta is delicious too!",
                    "user_id": "test_direct_user"
                }
            }
        }
        
        response = await client.post(f"{base_url}/mcp/", json=tool_payload_direct, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result}")
        else:
            print(f"   ❌ Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_mcp_2_0_format())