#!/usr/bin/env python3
"""
Test MCP 2.0 Parameter Structure with Proper Session Management
"""

import asyncio
import httpx
import json
import time

def parse_sse_response(response_text):
    """Parse SSE response to extract JSON data"""
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith('data: '):
            return json.loads(line[6:])
    return None

async def test_working_mcp_2_0():
    """Test MCP 2.0 with proper session management"""
    
    base_url = "http://localhost:5000"
    api_key = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    
    print("🧪 Testing Working MCP 2.0 Implementation")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    
    async with httpx.AsyncClient() as client:
        # 1. Initialize session properly
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "WorkingMCP2.0Client", "version": "1.0.0"}
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
            
            # Parse the SSE response
            init_result = parse_sse_response(response.text)
            if init_result and init_result.get('result'):
                print(f"   ✅ Initialization successful")
                
                # Wait for initialization to complete
                await asyncio.sleep(1)
                
                # 2. Send initialized message
                initialized_payload = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "method": "initialized",
                    "params": {}
                }
                
                print("\n2. Sending initialized notification...")
                response = await client.post(f"{base_url}/mcp/", json=initialized_payload, headers=headers)
                print(f"   Status: {response.status_code}")
                
                # Wait for server to process
                await asyncio.sleep(0.5)
                
                # 3. Test add_memory with MCP 2.0 parameter structure
                print("\n3. Testing add_memory with MCP 2.0 format...")
                tool_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "add_memory",
                        "arguments": {
                            "params": {  # MCP 2.0 nested structure
                                "user_input": "I love working with MCP protocols",
                                "agent_response": "MCP protocols are powerful for AI interactions!",
                                "user_id": "mcp_2_0_test_user"
                            }
                        }
                    }
                }
                
                response = await client.post(f"{base_url}/mcp/", json=tool_payload, headers=headers)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = parse_sse_response(response.text)
                    if result:
                        if 'result' in result:
                            print(f"   ✅ MCP 2.0 format successful!")
                            print(f"   Result: {result['result']}")
                        elif 'error' in result:
                            print(f"   ❌ Error: {result['error']}")
                    else:
                        print(f"   Raw response: {response.text}")
                else:
                    print(f"   ❌ Failed: {response.text}")
            else:
                print(f"   ❌ Initialization failed: {init_result}")
        else:
            print(f"   ❌ Connection failed: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_working_mcp_2_0())