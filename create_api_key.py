#!/usr/bin/env python3
"""
Generate and configure a valid API key for MemoryOS MCP Server
"""

import secrets
import os
import sys

def generate_valid_api_key():
    """Generate a cryptographically secure API key"""
    # Use the same method as the server (secrets.token_urlsafe)
    return secrets.token_urlsafe(32)

def main():
    print("🔑 Generating Valid API Key for MemoryOS MCP Server")
    print("=" * 60)
    
    # Generate a secure API key
    api_key = generate_valid_api_key()
    
    print(f"✅ Generated API Key: {api_key}")
    print(f"✅ Key Length: {len(api_key)} characters")
    print(f"✅ Key Type: URL-safe Base64 encoded")
    
    print("\n📋 How to Use This API Key:")
    print("1. Copy the API key above")
    print("2. Set it as environment variable:")
    print(f"   export MCP_API_KEY=\"{api_key}\"")
    print("3. Start the secure MCP server")
    print("4. Use the key in your MCP client")
    
    print(f"\n🧪 Test Commands:")
    print("Header-based authentication:")
    print(f"curl -X POST http://localhost:5000/mcp \\")
    print(f"  -H 'X-API-Key: {api_key}' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"id\":1}}'")
    
    print(f"\nBearer token authentication:")
    print(f"curl -X POST http://localhost:5000/mcp \\")
    print(f"  -H 'Authorization: Bearer {api_key}' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"id\":1}}'")
    
    return api_key

if __name__ == "__main__":
    main()