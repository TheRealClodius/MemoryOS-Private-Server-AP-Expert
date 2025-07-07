#!/usr/bin/env python3
"""
Setup API Key for MemoryOS MCP Server
"""

import os
import secrets
import subprocess
import sys

def generate_secure_key():
    """Generate a cryptographically secure API key"""
    return secrets.token_urlsafe(32)

def main():
    print("🔑 MemoryOS MCP Server - API Key Setup")
    print("=" * 50)
    
    # Generate a new secure key
    api_key = generate_secure_key()
    
    print(f"Generated API Key: {api_key}")
    print("\n📋 Setup Instructions:")
    print("1. Copy the API key above")
    print("2. Set it as environment variable (choose one method):")
    print(f"\n   Method A - Export (temporary):")
    print(f"   export MCP_API_KEY=\"{api_key}\"")
    print(f"\n   Method B - Add to .env file (permanent):")
    print(f"   echo 'MCP_API_KEY=\"{api_key}\"' >> .env")
    
    # Create .env file option
    create_env = input("\n❓ Would you like me to create a .env file with this key? (y/n): ").lower().strip()
    
    if create_env in ['y', 'yes']:
        with open('.env', 'a') as f:
            f.write(f'\nMCP_API_KEY="{api_key}"\n')
        print("✅ Added API key to .env file")
        print("⚠️  Restart your server to use the new key")
    
    print(f"\n🔐 How to use this API key:")
    print("Header method:")
    print(f"  curl -H 'X-API-Key: {api_key}' http://localhost:5000/api/...")
    print("Bearer token method:")
    print(f"  curl -H 'Authorization: Bearer {api_key}' http://localhost:5000/api/...")
    
    print(f"\n🧪 Test your API key:")
    print(f"curl -X POST http://localhost:5000/api/add_memory \\")
    print(f"  -H 'X-API-Key: {api_key}' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"user_id\":\"test\",\"user_input\":\"hello\",\"agent_response\":\"hi there!\"}}'")

if __name__ == "__main__":
    main()