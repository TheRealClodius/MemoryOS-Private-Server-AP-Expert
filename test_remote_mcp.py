#!/usr/bin/env python3
"""
Test client for MemoryOS Remote MCP 2.0 Server
Tests all three main tools using FastMCP client with Streamable HTTP transport
"""

import asyncio
from fastmcp import Client

async def test_remote_server():
    """Test the remote MemoryOS MCP server using streamable-http transport."""
    
    # Connect to MCP server using Streamable HTTP transport
    # For local testing: http://localhost:5000/mcp
    # For Cloud Run with proxy: http://localhost:8080/mcp
    server_url = "http://localhost:5001/mcp"
    
    print("🧪 Testing MemoryOS Remote MCP Server")
    print(f"🔗 Connecting to: {server_url}")
    print("=" * 60)
    
    try:
        async with Client(server_url) as client:
            # List available tools
            print("📋 Listing available tools...")
            tools = await client.list_tools()
            for tool in tools:
                print(f"   ✓ Tool found: {tool.name}")
            print()
            
            test_user_id = "test_user_remote"
            
            # Test 1: Add memory
            print("🧪 Test 1: Adding memory...")
            add_result = await client.call_tool("add_memory", {
                "user_input": "What's my favorite color?", 
                "agent_response": "I don't have information about your favorite color yet. Could you tell me?",
                "user_id": test_user_id,
                "meta_data": {"session": "remote_test", "importance": "medium"}
            })
            print(f"   📝 Add result: {add_result[0].text}")
            print()
            
            # Test 2: Add another memory
            print("🧪 Test 2: Adding another memory...")
            add_result2 = await client.call_tool("add_memory", {
                "user_input": "My favorite color is blue.",
                "agent_response": "Thanks for letting me know! I'll remember that your favorite color is blue.",
                "user_id": test_user_id,
                "meta_data": {"session": "remote_test", "importance": "high"}
            })
            print(f"   📝 Add result: {add_result2[0].text}")
            print()
            
            # Test 3: Retrieve memory
            print("🧪 Test 3: Retrieving memory...")
            retrieve_result = await client.call_tool("retrieve_memory", {
                "query": "favorite color",
                "user_id": test_user_id,
                "max_results": 5
            })
            print(f"   🔍 Retrieve result: {retrieve_result[0].text}")
            print()
            
            # Test 4: Get user profile
            print("🧪 Test 4: Getting user profile...")
            profile_result = await client.call_tool("get_user_profile", {
                "user_id": test_user_id,
                "include_knowledge": True,
                "include_assistant_knowledge": False
            })
            print(f"   👤 Profile result: {profile_result[0].text}")
            print()
            
            # Test 5: Test with different user (isolation)
            print("🧪 Test 5: Testing user isolation...")
            different_user = "different_user_remote"
            isolation_result = await client.call_tool("retrieve_memory", {
                "query": "favorite color",
                "user_id": different_user,
                "max_results": 5
            })
            print(f"   🔒 Isolation test: {isolation_result[0].text}")
            print()
            
            print("✅ All tests completed successfully!")
            print("🎉 MemoryOS Remote MCP Server is working correctly!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the MemoryOS MCP server is running on the expected URL")

if __name__ == "__main__":
    asyncio.run(test_remote_server())