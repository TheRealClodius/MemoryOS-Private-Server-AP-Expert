#!/usr/bin/env python3
"""
Test the actual production server at memory-os-private-server-ac.replit.app
"""

import asyncio
import httpx
from fastmcp import Client

async def test_production_endpoints():
    """Test various endpoints on the production server"""
    
    base_url = "https://memory-os-private-server-ac.replit.app"
    api_key = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    
    print("🌐 Testing Production Server Endpoints")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:8]}...")
    print()
    
    async with httpx.AsyncClient() as client:
        # Test 1: Health check
        print("1. Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 2: MCP endpoint without auth
        print("2. Testing MCP endpoint (no auth)...")
        try:
            response = await client.get(f"{base_url}/mcp/")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 3: MCP endpoint with Bearer auth
        print("3. Testing MCP endpoint (Bearer auth)...")
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.get(f"{base_url}/mcp/", headers=headers)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 4: MCP endpoint with X-API-Key auth
        print("4. Testing MCP endpoint (X-API-Key auth)...")
        try:
            headers = {"X-API-Key": api_key}
            response = await client.get(f"{base_url}/mcp/", headers=headers)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test 5: FastMCP Client
    print("5. Testing FastMCP Client...")
    try:
        async with Client(f"{base_url}/mcp/") as mcp_client:
            tools = await mcp_client.list_tools()
            print(f"   ✅ Connected! Found {len(tools)} tools")
            for tool in tools:
                print(f"      - {tool.name}")
    except Exception as e:
        print(f"   ❌ FastMCP failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_production_endpoints())