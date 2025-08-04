#!/usr/bin/env python3
"""
Deployment server for MemoryOS MCP Server
Simple and reliable deployment wrapper for production use
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add current directory to Python path to ensure imports work correctly
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Configure logging for deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main deployment entry point"""
    logger.info("Starting MemoryOS MCP Server deployment...")
    
    # Get port from environment (required for deployment)
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Deployment configured for port: {port}")
    
    # Set environment variable for consistency
    os.environ['PORT'] = str(port)
    
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment. Server may not function properly.")
    else:
        logger.info("OpenAI API key found in environment")
    
    # Ensure data directory exists
    data_dir = current_dir / "memoryos_data"
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory ready: {data_dir}")
    
    try:
        # Import and run the server
        import uvicorn
        from mcp_server import app
        
        logger.info("Starting MemoryOS Pure MCP 2.0 Server...")
        
        # Run the server with production settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()