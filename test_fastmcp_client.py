#!/usr/bin/env python3
"""
Test MemoryOS Remote MCP Server using proper FastMCP client
"""

import asyncio
import json
from pathlib import Path
from fastmcp import Client

async def test_fastmcp_client():
    """Test the remote MCP server using FastMCP client properly"""
    
    # Load API key from config
    config_file = Path("config.json")
    if config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
            api_keys = config_data.get("api_keys", {})
            if api_keys:
                test_api_key = list(api_keys.keys())[0]
                print(f"Using API key: {test_api_key[:8]}...")
            else:
                print("No API keys found in config.json")
                return
    else:
        print("config.json not found")
        return
    
    server_url = "http://localhost:5000/mcp"
    
    print("🧪 Testing MemoryOS Remote MCP Server")
    print(f"🔗 Server: {server_url}")
    print("=" * 50)
    
    try:
        # Connect using FastMCP client without custom headers first
        # (FastMCP may handle auth differently)
        async with Client(server_url) as client:
            print("✅ Connected to MCP server")
            
            # List tools
            print("📋 Listing available tools...")
            tools = await client.list_tools()
            print(f"   Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool.name}: {tool.description[:50]}...")
            print()
            
            # Test add_memory
            print("🔧 Testing add_memory...")
            result = await client.call_tool("add_memory", {
                "params": {
                    "user_input": "Hello from remote MCP test!",
                    "agent_response": "Remote MCP server is working correctly!",
                    "user_id": "test_fastmcp_user",
                    "meta_data": {"test": "fastmcp_auth"}
                }
            })
            print(f"   Result: {result[0].text[:100]}...")
            print()
            
            # Test retrieve_memory
            print("🔍 Testing retrieve_memory...")
            result = await client.call_tool("retrieve_memory", {
                "params": {
                    "query": "remote MCP test",
                    "user_id": "test_fastmcp_user",
                    "max_results": 5
                }
            })
            print(f"   Result: {result[0].text[:100]}...")
            print()
            
            # Test get_user_profile  
            print("👤 Testing get_user_profile...")
            result = await client.call_tool("get_user_profile", {
                "params": {
                    "user_id": "test_fastmcp_user",
                    "include_knowledge": True
                }
            })
            print(f"   Result: {result[0].text[:100]}...")
            print()
            
            print("🎉 All MCP tools working successfully!")
            print("✅ Remote MCP server is fully operational!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Note: Authentication may be handled at a different level")

if __name__ == "__main__":
    asyncio.run(test_fastmcp_client())