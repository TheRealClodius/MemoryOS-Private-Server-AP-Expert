#!/usr/bin/env python3
"""
MemoryOS Deployment Server - SECURITY FIXED VERSION
HTTP server for Replit deployment with proper user isolation and data security
"""

import os
import json
import asyncio
import signal
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import MemoryOS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memoryos.memoryos import Memoryos

async def startup_event():
    """Initialize default configuration on startup"""
    try:
        # Just load config to validate it
        config = load_config()
        print("MemoryOS server configuration loaded successfully", file=sys.stderr)
        print(f"Ready to serve requests with user isolation", file=sys.stderr)
    except Exception as e:
        print(f"Failed to load configuration: {e}", file=sys.stderr)
        raise

app = FastAPI(
    title="MemoryOS Server", 
    version="1.0.0",
    on_startup=[startup_event]
)

# Global configuration cache
_global_config = None
_user_instances = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with environment variable fallbacks"""
    global _global_config
    
    if _global_config is not None:
        return _global_config
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Override with environment variables
        config["openai_api_key"] = os.getenv("OPENAI_API_KEY", config.get("openai_api_key"))
        config["data_storage_path"] = os.getenv("MEMORYOS_DATA_PATH", config.get("data_storage_path", "./memoryos_data"))
        config["openai_base_url"] = os.getenv("OPENAI_BASE_URL", config.get("openai_base_url", "https://api.openai.com/v1"))
        config["llm_model"] = os.getenv("MEMORYOS_LLM_MODEL", config.get("llm_model", "gpt-4o-mini"))
        config["embedding_model"] = os.getenv("MEMORYOS_EMBEDDING_MODEL", config.get("embedding_model", "text-embedding-3-small"))
        
        # Memory configuration
        config.setdefault("short_term_capacity", int(os.getenv("MEMORYOS_SHORT_TERM_CAPACITY", "10")))
        config.setdefault("mid_term_capacity", int(os.getenv("MEMORYOS_MID_TERM_CAPACITY", "2000")))
        config.setdefault("long_term_knowledge_capacity", int(os.getenv("MEMORYOS_KNOWLEDGE_CAPACITY", "100")))
        config.setdefault("retrieval_queue_capacity", int(os.getenv("MEMORYOS_RETRIEVAL_CAPACITY", "7")))
        config.setdefault("mid_term_heat_threshold", float(os.getenv("MEMORYOS_HEAT_THRESHOLD", "5.0")))
        
        # Validate required fields
        if not config.get("openai_api_key"):
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or add to config.json")
        
        _global_config = config
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def get_memoryos_for_user(user_id: str, assistant_id: str = "mcp_assistant") -> Memoryos:
    """Get or create MemoryOS instance for specific user with proper isolation"""
    global _user_instances
    
    # Validate user_id
    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty or null")
    
    # Create cache key
    cache_key = f"{user_id}_{assistant_id}"
    
    # Return existing instance if available
    if cache_key in _user_instances:
        return _user_instances[cache_key]
    
    # Load base configuration
    config = load_config()
    
    # Create user-specific MemoryOS instance
    try:
        instance = Memoryos(
            user_id=user_id,
            assistant_id=assistant_id,
            openai_api_key=config.get("openai_api_key"),
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
        
        # Cache the instance
        _user_instances[cache_key] = instance
        print(f"Created new MemoryOS instance for user: {user_id}", file=sys.stderr)
        return instance
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MemoryOS for user {user_id}: {str(e)}")

@app.get("/")
async def health_check():
    """Health check endpoint for deployment"""
    try:
        # Test basic functionality with a temporary user
        test_user_id = f"health_check_{uuid.uuid4().hex[:8]}"
        test_instance = get_memoryos_for_user(test_user_id)
        
        return {
            "status": "healthy",
            "service": "MemoryOS MCP Server",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "user_isolation": "enabled",
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
        config = load_config()
        
        return {
            "status": "healthy",
            "service": "MemoryOS MCP Server", 
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "openai_configured": bool(config.get("openai_api_key")),
            "config_loaded": True,
            "user_isolation": "enabled",
            "security_features": [
                "user_data_isolation",
                "per_user_memory_instances",
                "user_id_validation"
            ]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "config_loaded": False
            }
        )

# Pydantic models for API requests
class AddMemoryRequest(BaseModel):
    user_id: str = Field(..., description="User identifier (required for data isolation)")
    user_input: str = Field(..., description="User's input or question")
    agent_response: str = Field(..., description="Assistant's response")
    timestamp: Optional[str] = Field(None, description="Optional timestamp in ISO format")
    meta_data: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    assistant_id: Optional[str] = Field("mcp_assistant", description="Assistant identifier")

@app.post("/api/add_memory")
async def add_memory_endpoint(request: AddMemoryRequest):
    """Add a new memory entry (HTTP endpoint) with user isolation"""
    try:
        # Validate user_id
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
        
        # Get user-specific MemoryOS instance
        server = get_memoryos_for_user(request.user_id, request.assistant_id)
        
        # Use provided timestamp or generate current one
        if request.timestamp is None:
            timestamp = datetime.now().isoformat()
        else:
            timestamp = request.timestamp
        
        # Add memory to the user's specific instance
        result = server.add_memory(request.user_input, request.agent_response, timestamp, request.meta_data)
        success = result.get("status") == "success"
        
        if success:
            return {
                "status": "success",
                "message": "Memory added successfully",
                "user_id": request.user_id,
                "timestamp": timestamp
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to add memory",
                    "user_id": request.user_id,
                    "timestamp": timestamp
                }
            )
    except HTTPException:
        raise
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
    user_id: str = Query(..., description="User identifier (required for data isolation)"),
    query: str = Query(..., description="Search query"),
    relationship_with_user: str = Query("assistant", description="Relationship context"),
    style_hint: str = Query("", description="Style hint for response"),
    max_results: int = Query(10, description="Maximum results to return"),
    assistant_id: str = Query("mcp_assistant", description="Assistant identifier")
):
    """Retrieve relevant memories (HTTP endpoint) with user isolation"""
    try:
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
        
        # Get user-specific MemoryOS instance
        server = get_memoryos_for_user(user_id, assistant_id)
        
        # Retrieve memories from the user's specific instance
        # This ensures only the user's data is searched - NO CROSS-USER DATA LEAKAGE
        result = server.retriever.retrieve_context(
            user_query=query,
            user_id=user_id  # This ensures only this user's data is retrieved
        )
        
        return {
            "status": "success",
            "query": query,
            "user_id": user_id,  # Include user_id in response for transparency
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/user_profile")
async def get_user_profile_endpoint(
    user_id: str = Query(..., description="User identifier (required for data isolation)"),
    include_knowledge: bool = Query(True, description="Include user knowledge"),
    include_assistant_knowledge: bool = Query(False, description="Include assistant knowledge"),
    assistant_id: str = Query("mcp_assistant", description="Assistant identifier")
):
    """Get user profile (HTTP endpoint) with user isolation"""
    try:
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
        
        # Get user-specific MemoryOS instance
        server = get_memoryos_for_user(user_id, assistant_id)
        
        # Get profile for the specific user only
        user_profile = server.get_user_profile_summary()
        
        return {
            "status": "success",
            "user_id": user_id,  # Include user_id in response for transparency
            "timestamp": datetime.now().isoformat(),
            "user_profile": user_profile
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )

class CreateUserRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier (auto-generated if not provided)")
    assistant_id: Optional[str] = Field("mcp_assistant", description="Assistant identifier")

@app.post("/api/create_user")
async def create_user_endpoint(request: CreateUserRequest):
    """Create a new user (HTTP endpoint) with proper isolation"""
    try:
        # Generate user_id if not provided
        if not request.user_id:
            request.user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Validate user_id
        if not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        # Create user-specific MemoryOS instance
        server = get_memoryos_for_user(request.user_id, request.assistant_id)
        
        # Get user data path for transparency
        config = load_config()
        user_data_path = f"{config['data_storage_path']}/{request.user_id}"
        
        return {
            "status": "success",
            "message": "User created successfully",
            "user_id": request.user_id,
            "assistant_id": request.assistant_id,
            "data_path": user_data_path,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/user_info")
async def get_user_info_endpoint(
    user_id: str = Query(..., description="User identifier")
):
    """Get current user information"""
    try:
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
        
        # Check if user exists
        config = load_config()
        user_data_path = f"{config['data_storage_path']}/{user_id}"
        
        # Get user-specific instance if it exists
        server = get_memoryos_for_user(user_id)
        stats = server.get_memory_stats()
        
        return {
            "status": "success",
            "user_id": user_id,
            "assistant_id": "mcp_assistant",
            "data_path": user_data_path,
            "memory_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
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
        # Set the deployment API key
        if not os.getenv("MCP_API_KEY"):
            os.environ["MCP_API_KEY"] = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
        
        # Import the remote MCP server
        import mcp_remote_server
        
        logger.info("Starting MemoryOS Remote MCP Server (StreamableHTTP)")
        logger.info(f"Server URL: http://0.0.0.0:{os.getenv('PORT', '5000')}")
        logger.info(f"API Key: {os.getenv('MCP_API_KEY')[:8]}...")
        logger.info("Transport: StreamableHTTP")
        logger.info("Ready for remote MCP clients with authentication")
        
        # Run the MCP server directly using its main function
        port = int(os.getenv("PORT", "5000"))
        await mcp_remote_server.mcp.run_async(
            transport="streamable-http",
            host="0.0.0.0", 
            port=port,
        )
        
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
    # Check if we should run as HTTP API server or MCP server
    if os.getenv("RUN_AS_API_SERVER", "false").lower() == "true":
        # Run as HTTP API server with user isolation
        port = int(os.getenv("PORT", 5000))
        print(f"Starting MemoryOS HTTP API server on port {port} with USER ISOLATION SECURITY", file=sys.stderr)
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Run as MCP server (default)
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
