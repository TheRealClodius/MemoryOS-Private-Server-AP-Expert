#!/usr/bin/env python3
"""
MemoryOS Deployment Server
Entry point for production deployment of the MemoryOS Pure MCP 2.0 Remote Server
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

# Import the main FastAPI app from mcp_server.py
from mcp_server import app, main

# Configure logging for deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_deployment_environment():
    """Setup environment variables and configuration for deployment"""
    
    # Ensure OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is required for deployment")
        # Don't exit - let the user know they need to provide the key
        logger.warning("Server will start but may not function properly without OpenAI API key")
    
    # Set default port if not specified
    if not os.getenv('PORT'):
        os.environ['PORT'] = '5000'
        logger.info("PORT not specified, defaulting to 5000")
    
    # Ensure data directory exists
    data_dir = Path("memoryos_data")
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory ensured at: {data_dir.absolute()}")
    
    # Log deployment configuration
    logger.info("Deployment environment configured:")
    logger.info(f"  - Port: {os.getenv('PORT', '5000')}")
    logger.info(f"  - OpenAI API Key: {'✓ Present' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
    logger.info(f"  - Data Directory: {data_dir.absolute()}")

def deploy_streamable_http():
    """Deploy the MemoryOS server for HTTP access (main deployment method)"""
    logger.info("Starting MemoryOS Pure MCP 2.0 deployment (HTTP mode)")
    
    # Setup environment
    setup_deployment_environment()
    
    # Run the main server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

def deploy_stdio():
    """Fallback deployment method - redirects to HTTP mode for compatibility"""
    logger.info("STDIO mode requested - redirecting to HTTP deployment for better compatibility")
    deploy_streamable_http()

if __name__ == "__main__":
    # Default to HTTP deployment
    deploy_streamable_http()