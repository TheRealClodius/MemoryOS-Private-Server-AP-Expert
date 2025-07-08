#!/usr/bin/env python3
"""
Main entry point for MemoryOS MCP Server deployment
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the current directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for deployment"""
    # Set your API key for deployment
    if not os.getenv("MCP_API_KEY"):
        os.environ["MCP_API_KEY"] = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    
    # Set port from environment
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting MemoryOS MCP Server on port {port}", file=sys.stderr)
    print(f"Using API Key: {os.getenv('MCP_API_KEY')[:8]}...", file=sys.stderr)
    
    # Import and run pure MCP 2.0 server for deployment
    from mcp_server import app
    
    # Run the MCP server with authentication
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()