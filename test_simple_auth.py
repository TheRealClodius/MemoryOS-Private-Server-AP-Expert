#!/usr/bin/env python3
"""
Simple test for API key authentication using direct HTTP requests
"""

import asyncio
import json
import httpx
from pathlib import Path

async def test_simple_auth():
    """Test API key authentication with direct HTTP requests"""
    
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
    
    server_url = "http://localhost:5000/mcp/"
    
    print("Testing API Key Authentication")
    print(f"Server: {server_url}")
    print("=" * 40)
    
    async with httpx.AsyncClient() as client:
        # Test 1: Valid API key
        print("Testing with valid API key...")
        headers = {
            "Authorization": f"Bearer {test_api_key}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        }
        
        response = await client.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            },
            headers=headers
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            tools = data.get("result", {}).get("tools", [])
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}")
        else:
            print(f"Error: {response.text}")
        
        print()
        
        # Test 2: Invalid API key
        print("Testing with invalid API key...")
        invalid_headers = {
            "Authorization": "Bearer invalid_key_123",
            "Accept": "application/json, text/event-stream", 
            "Content-Type": "application/json"
        }
        
        response = await client.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            },
            headers=invalid_headers
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 401:
            print("Expected 401 unauthorized - authentication working!")
        else:
            print(f"Unexpected response: {response.text}")
        
        print()
        
        # Test 3: Test add_memory tool
        print("Testing add_memory tool...")
        response = await client.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "add_memory",
                    "arguments": {
                        "user_input": "Testing remote MCP authentication",
                        "agent_response": "Authentication is working properly!",
                        "user_id": "test_auth_user_simple"
                    }
                }
            },
            headers=headers
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Memory added successfully!")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_simple_auth())