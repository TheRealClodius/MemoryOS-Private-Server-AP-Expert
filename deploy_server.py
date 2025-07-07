#!/usr/bin/env python3
"""
MemoryOS Deployment Server
HTTP server for Replit deployment with health checks and MCP functionality
"""

import sys
import os
import json
import asyncio
import signal
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    print(f"ERROR: Failed to import FastAPI/uvicorn. Please install: pip install fastapi uvicorn", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from memoryos import Memoryos
except ImportError as e:
    print(f"ERROR: Failed to import MemoryOS. Ensure memoryos package is available.", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

# Global MemoryOS instance
memoryos_instance = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with environment variable fallbacks"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Override with environment variables if available
        if os.getenv("OPENAI_API_KEY"):
            config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_BASE_URL"):
            config["openai_base_url"] = os.getenv("OPENAI_BASE_URL")
        
        # Validate required fields
        required_fields = ["user_id", "assistant_id", "llm_model", "embedding_model"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def init_memoryos(config: Dict[str, Any]) -> Memoryos:
    """Initialize MemoryOS instance with configuration"""
    try:
        return Memoryos(
            user_id=config["user_id"],
            assistant_id=config["assistant_id"],
            openai_api_key=config.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
            openai_base_url=config.get("openai_base_url", "https://api.openai.com/v1"),
            data_storage_path=config.get("data_storage_path", "./memoryos_data"),
            short_term_capacity=config.get("short_term_capacity", 10),
            mid_term_capacity=config.get("mid_term_capacity", 2000),
            long_term_knowledge_capacity=config.get("long_term_knowledge_capacity", 100),
            retrieval_queue_capacity=config.get("retrieval_queue_capacity", 7),
            mid_term_heat_threshold=config.get("mid_term_heat_threshold", 5.0),
            llm_model=config["llm_model"],
            embedding_model=config["embedding_model"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MemoryOS: {str(e)}")

async def init_server():
    """Initialize the MemoryOS server"""
    global memoryos_instance
    
    if memoryos_instance is None:
        print("Loading MemoryOS configuration...", file=sys.stderr)
        config = load_config()
        
        print(f"Initializing MemoryOS for user: {config['user_id']}", file=sys.stderr)
        memoryos_instance = init_memoryos(config)
        
        print(f"MemoryOS initialized successfully", file=sys.stderr)
        print(f"User: {config['user_id']}, Assistant: {config['assistant_id']}", file=sys.stderr)
        print(f"Data storage: {config['data_storage_path']}", file=sys.stderr)
        print(f"LLM model: {config['llm_model']}, Embedding model: {config['embedding_model']}", file=sys.stderr)
    
    return memoryos_instance

# Create FastAPI app
app = FastAPI(
    title="MemoryOS MCP Server",
    description="Memory Operating System for Personalized AI Agents",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize MemoryOS on startup"""
    try:
        await init_server()
        print("MemoryOS server started successfully", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize MemoryOS: {e}", file=sys.stderr)
        # Don't exit here - let health checks handle the error

@app.get("/")
async def health_check():
    """Health check endpoint for deployment"""
    try:
        # Initialize server if not already done
        await init_server()
        
        return {
            "status": "healthy",
            "service": "MemoryOS MCP Server",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "capabilities": [
                "add_memory",
                "retrieve_memory", 
                "get_user_profile"
            ]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health")
async def detailed_health():
    """Detailed health check"""
    try:
        server = await init_server()
        
        # Test basic functionality
        stats = server.get_memory_stats()
        
        return {
            "status": "healthy",
            "service": "MemoryOS MCP Server",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "memory_stats": stats,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "config_loaded": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
                "config_loaded": False
            }
        )

from pydantic import BaseModel

class AddMemoryRequest(BaseModel):
    user_input: str
    agent_response: str
    timestamp: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None

@app.post("/api/add_memory")
async def add_memory_endpoint(request: AddMemoryRequest):
    """Add a new memory entry (HTTP endpoint)"""
    try:
        server = await init_server()
        
        # Use provided timestamp or generate current one
        if request.timestamp is None:
            timestamp = datetime.now().isoformat()
        else:
            timestamp = request.timestamp
        
        # Add memory to the system
        result = server.add_memory(request.user_input, request.agent_response, timestamp, request.meta_data)
        success = result.get("status") == "success"
        
        if success:
            return {
                "status": "success",
                "message": "Memory added successfully",
                "timestamp": timestamp
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to add memory",
                    "timestamp": timestamp
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/retrieve_memory")
async def retrieve_memory_endpoint(
    query: str,
    relationship_with_user: str = "assistant",
    style_hint: str = "",
    max_results: int = 10
):
    """Retrieve relevant memories (HTTP endpoint)"""
    try:
        server = await init_server()
        
        # Retrieve memories from the system
        result = server.retriever.retrieve_context(
            user_query=query,
            user_id=server.user_id
        )
        
        return {
            "status": "success",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/user_profile")
async def get_user_profile_endpoint():
    """Get user profile (HTTP endpoint)"""
    try:
        server = await init_server()
        
        # Get user profile from the system
        profile = server.get_user_profile_summary()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_profile": profile
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

async def deploy_streamable_http():
    """Deploy MemoryOS as a StreamableHTTP MCP server"""
    try:
        # Import after path setup
        from mcp_server import run_streamable_http_server
        
        logger.info("Starting MemoryOS MCP Server (StreamableHTTP)")
        logger.info(f"Server URL: http://0.0.0.0:{os.getenv('PORT', '3000')}")
        logger.info("Transport: StreamableHTTP")
        logger.info("Ready for remote MCP clients")
        
        # Run the server
        await run_streamable_http_server()
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

async def deploy_stdio():
    """Deploy MemoryOS as a stdio MCP server (legacy mode)"""
    try:
        # Import after path setup
        from mcp_server import mcp
        
        logger.info("Starting MemoryOS MCP Server (stdio - legacy mode)")
        logger.info("Transport: stdio")
        logger.info("Ready for local MCP clients")
        
        # Run the server
        await mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

async def main():
    """Main deployment entry point"""
    # Check deployment mode
    mode = os.getenv("SERVER_MODE", "streamable-http").lower()
    port = os.getenv("PORT", "3000")
    
    logger.info(f"MemoryOS MCP Server Deployment")
    logger.info(f"Mode: {mode}")
    logger.info(f"Port: {port}")
    
    if mode == "stdio":
        await deploy_stdio()
    elif mode in ["streamable-http", "http"]:
        await deploy_streamable_http()
    else:
        logger.error(f"Unknown deployment mode: {mode}")
        logger.error("Available modes: stdio, streamable-http, http")
        sys.exit(1)

if __name__ == "__main__":
    # Set default to StreamableHTTP for deployment
    if "SERVER_MODE" not in os.environ:
        os.environ["SERVER_MODE"] = "streamable-http"
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Deployment stopped by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)