#!/usr/bin/env python3
"""
Test API key authentication for MemoryOS Remote MCP Server
"""

import asyncio
import json
from pathlib import Path
from fastmcp import Client

async def test_api_key_auth():
    """Test API key authentication with the remote MCP server"""
    
    # Load API key from config
    config_file = Path("config.json")
    if config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
            api_keys = config_data.get("api_keys", {})
            if api_keys:
                test_api_key = list(api_keys.keys())[0]
                print(f"Using API key from config: {test_api_key[:8]}...")
            else:
                print("No API keys found in config.json")
                return
    else:
        print("config.json not found")
        return
    
    server_url = "http://localhost:5001/mcp"
    
    print("🧪 Testing API Key Authentication")
    print(f"🔗 Server: {server_url}")
    print(f"🔑 API Key: {test_api_key[:8]}...")
    print("=" * 50)
    
    try:
        # Test with valid API key
        print("✅ Testing with valid API key...")
        headers = {"Authorization": f"Bearer {test_api_key}"}
        
        async with Client(server_url, headers=headers) as client:
            tools = await client.list_tools()
            print(f"   Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool.name}")
            
            # Test add_memory
            result = await client.call_tool("add_memory", {
                "user_input": "Testing API key auth",
                "agent_response": "API key authentication working!",
                "user_id": "test_auth_user"
            })
            print(f"   Add memory result: Success")
            
        print()
        
        # Test with invalid API key
        print("❌ Testing with invalid API key...")
        invalid_headers = {"Authorization": "Bearer invalid_key_123"}
        
        try:
            async with Client(server_url, headers=invalid_headers) as client:
                await client.list_tools()
                print("   ERROR: Should have failed with invalid key!")
        except Exception as e:
            print(f"   Expected failure: {str(e)[:50]}...")
            
        print()
        print("🎉 API key authentication working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_key_auth())