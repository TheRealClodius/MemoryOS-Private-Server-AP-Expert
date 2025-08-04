#!/usr/bin/env python3
"""
MemoryOS API Key Manager
Generates and manages API keys for MemoryOS MCP Server
"""

import secrets
import os
import sys

def generate_api_key(length=32):
    """Generate a cryptographically secure URL-safe API key"""
    return secrets.token_urlsafe(length)

def create_env_file(api_key):
    """Create or append API key to .env file"""
    try:
        with open('.env', 'a') as f:
            f.write(f'\nMCP_API_KEY="{api_key}"\n')
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def print_usage_instructions(api_key):
    """Print comprehensive usage instructions"""
    print(f"\n📋 How to Use This API Key:")
    print("1. Copy the API key above")
    print("2. Set it as environment variable:")
    print(f"   export MCP_API_KEY=\"{api_key}\"")
    print("3. Start the MCP server")
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

def main():
    print("🔑 MemoryOS API Key Manager")
    print("=" * 50)
    
    # Generate a secure API key
    api_key = generate_api_key(32)
    
    print(f"✅ Generated API Key: {api_key}")
    print(f"✅ Key Length: {len(api_key)} characters")
    print(f"✅ Key Type: URL-safe Base64 encoded")
    
    # Ask about .env file creation
    create_env = input("\n❓ Would you like me to create a .env file with this key? (y/n): ").lower().strip()
    
    if create_env in ['y', 'yes']:
        if create_env_file(api_key):
            print("✅ Added API key to .env file")
            print("⚠️  Restart your server to use the new key")
    
    # Print usage instructions
    print_usage_instructions(api_key)
    
    return api_key

if __name__ == "__main__":
    main()