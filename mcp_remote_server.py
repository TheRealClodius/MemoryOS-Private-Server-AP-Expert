#!/usr/bin/env python3
"""
MemoryOS Remote MCP 2.0 Server using FastMCP
Aligned with MCP 2.0 JSON-RPC specification for nested parameter structure
Compatible with client MCP 2.0 implementations using "params" wrapper
"""

import asyncio
import json
import logging
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import our MemoryOS components
from memoryos.memoryos import Memoryos

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

# Initialize FastMCP server with standard MCP protocol compliance
mcp = FastMCP("MemoryOS Remote MCP Server")

# MCP 2.0 Parameter Extraction Helper
def extract_mcp_2_0_params(arguments: dict) -> dict:
    """Extract parameters from MCP 2.0 nested structure or direct format"""
    if "params" in arguments:
        # MCP 2.0 client format: {"arguments": {"params": {...}}}
        return arguments["params"]
    else:
        # Direct format: {"arguments": {...}}
        return arguments

# Custom tool wrapper for MCP 2.0 compatibility
def mcp_2_0_tool(func):
    """Decorator to handle MCP 2.0 parameter extraction"""
    def wrapper(**kwargs):
        # Extract parameters from MCP 2.0 structure if needed
        if len(kwargs) == 1 and "arguments" in kwargs:
            actual_params = extract_mcp_2_0_params(kwargs["arguments"])
            return func(**actual_params)
        else:
            return func(**kwargs)
    return wrapper

# Load API keys from environment or config
def load_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables or config file"""
    api_keys = {}
    
    # Try to load from config file
    config_file = Path("config.json")
    if config_file.exists():
        try:
            with open(config_file) as f:
                config_data = json.load(f)
                api_keys.update(config_data.get("api_keys", {}))
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
    
    # Load from environment variables (priority override)
    env_api_key = os.getenv("MCP_API_KEY")
    if env_api_key:
        api_keys = {env_api_key: {"user": "env_user", "description": "Environment API key"}}
        logger.info(f"Using environment API key: {env_api_key[:8]}...")
    
    # Generate default key if none exist
    if not api_keys:
        default_key = secrets.token_urlsafe(32)
        api_keys[default_key] = {"user": "default_user", "description": "Auto-generated default key"}
        logger.warning(f"Generated default API key: {default_key}")
        
        # Save to config file
        config_data = {"api_keys": api_keys}
        with open("config.json", "w") as f:
            json.dump(config_data, f, indent=2)
    
    return {key: info for key, info in api_keys.items()}

# Load API keys
valid_api_keys = load_api_keys()

# Global configuration
config = {
    "data_storage_path": "./memoryos_data",
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "short_term_capacity": 10,
    "mid_term_capacity": 2000,
    "enable_mid_term_overflow": True,
    "heat_threshold": 3,
    "use_faiss": True
}

# User memory instances cache for proper isolation
user_memory_instances: Dict[str, Memoryos] = {}

def get_memoryos_for_user(user_id: str) -> Memoryos:
    """Get or create MemoryOS instance for specific user with proper isolation"""
    if user_id not in user_memory_instances:
        logger.info(f"Creating new MemoryOS instance for user: {user_id}")
        
        # Create user-specific data path
        user_data_path = Path(config["data_storage_path"]) / user_id
        user_data_path.mkdir(parents=True, exist_ok=True)
        
        # Create user-specific config
        user_config = config.copy()
        user_config["data_storage_path"] = str(user_data_path)
        
        # Initialize MemoryOS instance with correct parameter mapping
        user_memory_instances[user_id] = Memoryos(
            user_id=user_id,
            openai_api_key=user_config["openai_api_key"],
            data_storage_path=user_config["data_storage_path"],
            llm_model=user_config["llm_model"],
            embedding_model=user_config["embedding_model"],
            short_term_capacity=user_config["short_term_capacity"],
            mid_term_capacity=user_config["mid_term_capacity"]
        )
        
    return user_memory_instances[user_id]

# Input Models for MCP 2.0 structured parameters
class AddMemoryInput(BaseModel):
    user_input: str = Field(description="The user's input or question")
    agent_response: str = Field(description="The agent's response")
    user_id: str = Field(description="The user identifier for memory isolation (required)")
    timestamp: Optional[str] = Field(None, description="Optional timestamp in ISO format")
    meta_data: Optional[Dict[str, Any]] = Field(None, description="Optional metadata dictionary")

class RetrieveMemoryInput(BaseModel):
    query: str = Field(description="Search query")
    user_id: str = Field(description="The user identifier for memory isolation (required)")
    relationship_with_user: str = Field("assistant", description="Relationship context")
    style_hint: str = Field("", description="Style preference")
    max_results: int = Field(10, description="Maximum number of results")

class GetUserProfileInput(BaseModel):
    user_id: str = Field(description="The user identifier for profile retrieval (required)")
    include_knowledge: bool = Field(True, description="Whether to include user knowledge entries")
    include_assistant_knowledge: bool = Field(False, description="Whether to include assistant knowledge entries")

@mcp.tool()
def add_memory(user_input: str, agent_response: str, user_id: str, timestamp: str = None, meta_data: dict = None) -> dict:
    """
    Add a new memory entry to the MemoryOS system with user isolation.
    
    Args:
        user_input: The user's input or question
        agent_response: The agent's response
        user_id: The user identifier for memory isolation (required)
        timestamp: Optional timestamp in ISO format
        meta_data: Optional metadata dictionary
    
    Returns:
        Structured result with operation status and details
    """
    try:
        logger.info(f">>> Tool: 'add_memory' called for user '{user_id}'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Use provided timestamp or current time
        timestamp = timestamp or datetime.now().isoformat()
        
        # Add memory to user's MemoryOS
        memoryos_instance.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp,
            meta_data=meta_data or {}
        )
        
        return {
            "status": "success",
            "message": "Memory added successfully",
            "timestamp": timestamp,
            "user_id": user_id,
            "details": {
                "user_input_length": len(user_input),
                "agent_response_length": len(agent_response),
                "has_metadata": bool(meta_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error adding memory for user {user_id}: {e}")
        return {
            "status": "error",
            "message": f"Failed to add memory: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }

@mcp.tool()
def retrieve_memory(query: str, user_id: str, relationship_with_user: str = "assistant", style_hint: str = "", max_results: int = 10) -> dict:
    """
    Retrieve relevant memories from MemoryOS system with user isolation.
    
    Args:
        query: Search query
        user_id: The user identifier for memory isolation (required)
        relationship_with_user: Relationship context
        style_hint: Style preference
        max_results: Maximum number of results
    
    Returns:
        Comprehensive memory context with relevant entries from all memory tiers
    """
    try:
        logger.info(f">>> Tool: 'retrieve_memory' called for user '{user_id}' with query: '{query[:50]}...'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Retrieve memories using MemoryOS
        result = memoryos_instance.retriever.retrieve_context(
            user_query=query,
            user_id=user_id
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error retrieving memory for user {user_id}: {e}")
        return {
            "status": "error",
            "message": f"Failed to retrieve memory: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query
        }

@mcp.tool()
def get_user_profile(user_id: str, include_knowledge: bool = True, include_assistant_knowledge: bool = False) -> dict:
    """
    Get comprehensive user profile and knowledge information with user isolation.
    
    Retrieves user profile analysis based on conversation history, including personality traits,
    preferences, and optionally associated knowledge entries.
    
    Args:
        params: GetUserProfileInput containing user_id and knowledge inclusion flags
    
    Returns:
        User profile with optional knowledge entries
    """
    try:
        logger.info(f">>> Tool: 'get_user_profile' called for user '{params.user_id}'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(params.user_id)
        
        # Get user profile
        profile = memoryos_instance.user_long_term_memory.get_user_profile_summary()
        
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": params.user_id,
            "user_profile": profile
        }
        
        # Include user knowledge if requested
        if params.include_knowledge:
            user_knowledge = memoryos_instance.user_long_term_memory.get_user_knowledge()
            result["user_knowledge"] = user_knowledge
            result["user_knowledge_count"] = len(user_knowledge)
        
        # Include assistant knowledge if requested
        if params.include_assistant_knowledge:
            assistant_knowledge = memoryos_instance.user_long_term_memory.get_assistant_knowledge()
            result["assistant_knowledge"] = assistant_knowledge
            result["assistant_knowledge_count"] = len(assistant_knowledge)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting user profile for {params.user_id}: {e}")
        return {
            "status": "error",
            "message": f"Failed to get user profile: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": params.user_id
        }

# Export app for deployment using FastMCP's internal app creation
def create_deployment_app():
    """Create FastAPI app for deployment"""
    import asyncio
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create a simple FastAPI wrapper that runs the MCP server
    app = FastAPI(title="MemoryOS Remote MCP Server")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create MCP transport and mount it
    @app.on_event("startup")
    async def startup():
        # The MCP server will run in the background
        pass
    
    # Health check endpoint
    @app.get("/")
    async def health_check():
        return {
            "status": "healthy",
            "service": "MemoryOS Remote MCP Server",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
    
    # Mount MCP server at /mcp/ endpoint
    from fastapi import Request
    
    @app.api_route("/mcp/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def mcp_proxy(request: Request, path: str):
        # This is a simple proxy - in production you'd want proper MCP handling
        return {"message": "MCP endpoint - use proper MCP client"}
    
    return app

# Create app for deployment
app = create_deployment_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"🚀 MemoryOS Remote MCP Server starting on port {port}")
    logger.info(f"🔧 Configuration: {config['llm_model']} + {config['embedding_model']}")
    logger.info(f"📁 Data storage: {config['data_storage_path']}")
    logger.info(f"🔑 Loaded {len(valid_api_keys)} API key(s)")
    
    # Show API keys for development (remove in production)
    for key, info in valid_api_keys.items():
        logger.info(f"   API Key: {key[:8]}... (user: {info.get('user', 'unknown')})")
    
    # Create custom app with authentication
    from fastapi import FastAPI, HTTPException, Security, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    
    # Custom authentication function
    security = HTTPBearer()
    
    async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
        if credentials.credentials not in valid_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return credentials.credentials
    
    # Run with streamable-http transport (MCP 2.0 standard)
    # host="0.0.0.0" required for Cloud Run and Docker deployments
    asyncio.run(
        mcp.run_async(
            transport="streamable-http",
            host="0.0.0.0", 
            port=port,
        )
    )