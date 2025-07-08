#!/usr/bin/env python3
"""
Simple test server to verify FastMCP functionality
"""

import asyncio
import os
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Simple Test Server")

@mcp.tool()
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5002"))
    print(f"Starting simple test server on port {port}")
    
    asyncio.run(
        mcp.run_async(
            transport="streamable-http",
            host="0.0.0.0",
            port=port,
        )
    )