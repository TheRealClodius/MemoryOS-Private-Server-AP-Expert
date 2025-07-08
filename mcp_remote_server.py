#!/usr/bin/env python3
"""
MemoryOS Remote MCP 2.0 Server using FastMCP
Fully compliant with MCP 2.0 Streamable HTTP Transport specification
Based on Google Cloud Run MCP deployment guidelines
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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Import our MemoryOS components
from memoryos.memoryos import Memoryos

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

# API Key Authentication
class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, valid_api_keys: Dict[str, str]):
        super().__init__(app)
        self.valid_api_keys = valid_api_keys
        
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks and docs
        if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
            return await call_next(request)
            
        # Extract API key from headers
        api_key = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        elif "x-api-key" in request.headers:
            api_key = request.headers["x-api-key"]
            
        # Validate API key
        if not api_key or api_key not in self.valid_api_keys:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"}
            )
            
        # Add user info to request state
        request.state.api_key_info = self.valid_api_keys[api_key]
        return await call_next(request)

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
    
    # Load from environment variables
    env_api_key = os.getenv("MCP_API_KEY")
    if env_api_key:
        api_keys[env_api_key] = {"user": "env_user", "description": "Environment API key"}
    
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

# Initialize FastMCP server 
mcp = FastMCP("MemoryOS Remote MCP Server")

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
        
        # Initialize MemoryOS instance
        user_memory_instances[user_id] = Memoryos(config_path=None, config_dict=user_config)
        
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
def add_memory(params: AddMemoryInput) -> Dict[str, Any]:
    """
    Add a new memory entry to the MemoryOS system with user isolation.
    
    Stores conversation pairs (user input + agent response) in the hierarchical memory system
    for building persistent dialogue history and contextual understanding.
    
    Args:
        params: AddMemoryInput containing user_input, agent_response, user_id, optional timestamp and metadata
    
    Returns:
        Structured result with operation status and details
    """
    try:
        logger.info(f">>> Tool: 'add_memory' called for user '{params.user_id}'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(params.user_id)
        
        # Use provided timestamp or current time
        timestamp = params.timestamp or datetime.now().isoformat()
        
        # Add memory to user's MemoryOS
        memoryos_instance.add_memory(
            user_input=params.user_input,
            agent_response=params.agent_response,
            timestamp=timestamp,
            meta_data=params.meta_data or {}
        )
        
        return {
            "status": "success",
            "message": "Memory added successfully",
            "timestamp": timestamp,
            "user_id": params.user_id,
            "details": {
                "user_input_length": len(params.user_input),
                "agent_response_length": len(params.agent_response),
                "has_metadata": bool(params.meta_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error adding memory for user {params.user_id}: {e}")
        return {
            "status": "error",
            "message": f"Failed to add memory: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": params.user_id
        }

@mcp.tool()
def retrieve_memory(params: RetrieveMemoryInput) -> Dict[str, Any]:
    """
    Retrieve relevant memories from MemoryOS system with user isolation.
    
    Searches across all memory layers (short-term, mid-term, long-term) using semantic similarity
    to find relevant conversation history, user knowledge, and assistant knowledge.
    
    Args:
        params: RetrieveMemoryInput containing query, user_id, and search parameters
    
    Returns:
        Comprehensive memory context with relevant entries from all memory tiers
    """
    try:
        logger.info(f">>> Tool: 'retrieve_memory' called for user '{params.user_id}' with query: '{params.query[:50]}...'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(params.user_id)
        
        # Retrieve memories using MemoryOS
        result = memoryos_instance.retrieve_memory(
            query=params.query,
            relationship_with_user=params.relationship_with_user,
            style_hint=params.style_hint,
            max_results=params.max_results
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": params.user_id,
            "query": params.query,
            "user_profile": result.get("user_profile", ""),
            "short_term_memory": result.get("short_term_memory", []),
            "retrieved_pages": result.get("retrieved_pages", []),
            "retrieved_user_knowledge": result.get("retrieved_user_knowledge", []),
            "retrieved_assistant_knowledge": result.get("retrieved_assistant_knowledge", []),
            "counts": {
                "short_term": len(result.get("short_term_memory", [])),
                "retrieved_pages": len(result.get("retrieved_pages", [])),
                "user_knowledge": len(result.get("retrieved_user_knowledge", [])),
                "assistant_knowledge": len(result.get("retrieved_assistant_knowledge", []))
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving memory for user {params.user_id}: {e}")
        return {
            "status": "error",
            "message": f"Failed to retrieve memory: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": params.user_id,
            "query": params.query
        }

@mcp.tool()
def get_user_profile(params: GetUserProfileInput) -> Dict[str, Any]:
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
        profile = memoryos_instance.get_user_profile_summary()
        
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