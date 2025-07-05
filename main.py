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
    # Import the FastAPI app
    from deploy_server import app
    
    # Get port from environment
    port = int(os.getenv("PORT", "5000"))
    
    print(f"Starting MemoryOS MCP Server on port {port}", file=sys.stderr)
    print(f"Health check endpoint: http://0.0.0.0:{port}/", file=sys.stderr)
    print(f"API endpoints: http://0.0.0.0:{port}/api/", file=sys.stderr)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()