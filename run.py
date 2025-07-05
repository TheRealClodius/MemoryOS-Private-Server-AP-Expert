#!/usr/bin/env python3
"""
MemoryOS Run Script
Determines whether to run HTTP server (for deployment) or MCP server (for local use)
"""

import os
import sys
import asyncio

def main():
    """Main entry point"""
    # Check if we're in deployment mode (PORT environment variable set)
    if os.getenv("PORT"):
        print("Running in deployment mode (HTTP server)", file=sys.stderr)
        # Run the deployment server directly
        import uvicorn
        from deploy_server import app
        
        port = int(os.getenv("PORT", "5000"))
        print(f"Starting HTTP server on port {port}", file=sys.stderr)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    else:
        print("Running in local mode (MCP server)", file=sys.stderr)
        # Import and run the MCP server
        import mcp_server
        asyncio.run(mcp_server.main())

if __name__ == "__main__":
    main()