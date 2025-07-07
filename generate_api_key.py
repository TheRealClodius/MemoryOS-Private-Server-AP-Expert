#!/usr/bin/env python3
"""
API Key Generator for MemoryOS MCP Server
Generates cryptographically secure API keys
"""

import secrets
import string

def generate_api_key(length=32):
    """Generate a secure API key"""
    # Generate URL-safe base64 encoded key
    return secrets.token_urlsafe(length)

def generate_hex_key(length=32):
    """Generate a hexadecimal API key"""
    return secrets.token_hex(length)

def generate_alphanumeric_key(length=32):
    """Generate an alphanumeric API key"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

if __name__ == "__main__":
    print("🔑 MemoryOS MCP Server - API Key Generator")
    print("=" * 50)
    
    # Generate different types of keys
    url_safe_key = generate_api_key(32)
    hex_key = generate_hex_key(16)
    alpha_key = generate_alphanumeric_key(32)
    
    print(f"URL-Safe Key (Recommended): {url_safe_key}")
    print(f"Hex Key:                    {hex_key}")
    print(f"Alphanumeric Key:           {alpha_key}")
    
    print("\n📋 Usage Instructions:")
    print("1. Copy one of the keys above")
    print("2. Set it as an environment variable:")
    print(f"   export MCP_API_KEY=\"{url_safe_key}\"")
    print("3. Restart your MCP server")
    print("4. Use the key in your API requests")
    
    print("\n🔐 Authentication Examples:")
    print("Header-based:")
    print(f"   X-API-Key: {url_safe_key}")
    print("Bearer token:")
    print(f"   Authorization: Bearer {url_safe_key}")