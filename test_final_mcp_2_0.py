#!/usr/bin/env python3
"""
Final test for MCP 2.0 compatibility with proper parameter handling
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

async def test_final_mcp_2_0():
    """Test final MCP 2.0 implementation"""
    
    base_url = "http://localhost:5000"
    api_key = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    
    print("🚀 Final MCP 2.0 Compatibility Test")
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
                "clientInfo": {"name": "FinalMCP2.0Client", "version": "1.0.0"}
            }
        }
        
        print("1. Initializing MCP session...")
        response = await client.post(f"{base_url}/mcp/", json=init_payload, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            session_id = response.headers.get('mcp-session-id')
            if session_id:
                headers["MCP-Session-ID"] = session_id
                print(f"   Session ID: {session_id}")
            
            # Parse initialization response
            init_result = parse_sse_response(response.text)
            if init_result and init_result.get('result'):
                print(f"   ✅ Initialization successful")
                
                # 2. Send initialized notification
                initialized_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                
                print("\n2. Sending initialized notification...")
                response = await client.post(f"{base_url}/mcp/", json=initialized_payload, headers=headers)
                print(f"   Status: {response.status_code}")
                
                await asyncio.sleep(1)
                
                # 3. Test direct parameter format first
                print("\n3. Testing direct parameter format...")
                direct_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "add_memory",
                        "arguments": {
                            "user_input": "I love direct parameter format",
                            "agent_response": "Direct parameters work great!",
                            "user_id": "direct_test_user"
                        }
                    }
                }
                
                response = await client.post(f"{base_url}/mcp/", json=direct_payload, headers=headers)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = parse_sse_response(response.text)
                    if result and 'result' in result:
                        print(f"   ✅ Direct format successful!")
                        print(f"   Result: {result['result']}")
                    elif result and 'error' in result:
                        print(f"   ❌ Direct format error: {result['error']}")
                
                # 4. Test MCP 2.0 client format
                print("\n4. Testing MCP 2.0 client format...")
                mcp2_payload = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "add_memory",
                        "arguments": {
                            "params": {  # MCP 2.0 nested structure
                                "user_input": "I love MCP 2.0 format",
                                "agent_response": "MCP 2.0 format should work too!",
                                "user_id": "mcp2_test_user"
                            }
                        }
                    }
                }
                
                response = await client.post(f"{base_url}/mcp/", json=mcp2_payload, headers=headers)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = parse_sse_response(response.text)
                    if result and 'result' in result:
                        print(f"   ✅ MCP 2.0 format successful!")
                        print(f"   Result: {result['result']}")
                    elif result and 'error' in result:
                        print(f"   ❌ MCP 2.0 format error: {result['error']}")
                
                # 5. Test retrieve_memory
                print("\n5. Testing retrieve_memory...")
                retrieve_payload = {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "retrieve_memory",
                        "arguments": {
                            "query": "programming languages",
                            "user_id": "direct_test_user"
                        }
                    }
                }
                
                response = await client.post(f"{base_url}/mcp/", json=retrieve_payload, headers=headers)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = parse_sse_response(response.text)
                    if result and 'result' in result:
                        print(f"   ✅ Retrieve memory successful!")
                        print(f"   Result: {result['result']}")
                    elif result and 'error' in result:
                        print(f"   ❌ Retrieve error: {result['error']}")
            else:
                print(f"   ❌ Initialization failed: {init_result}")
        else:
            print(f"   ❌ Connection failed: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_final_mcp_2_0())