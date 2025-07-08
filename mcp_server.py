#!/usr/bin/env python3
"""
MemoryOS Pure MCP 2.0 Remote Server
Direct implementation of MCP 2.0 JSON-RPC specification without framework dependencies
Fully compliant with official MCP 2.0 protocol standards
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

# Import our MemoryOS components
from memoryos.memoryos import Memoryos

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

# Security
security = HTTPBearer()

# Configuration and setup
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with environment variable fallbacks"""
    default_config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "data_storage_path": "./memoryos_data",
        "short_term_capacity": 7,
        "mid_term_capacity": 2000,
        "long_term_knowledge_capacity": 100,
        "retrieval_queue_capacity": 7,
        "mid_term_heat_threshold": 5.0,
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small"
    }
    
    try:
        with open(config_path, "r") as f:
            file_config = json.load(f)
            default_config.update(file_config)
    except FileNotFoundError:
        logger.info(f"Config file {config_path} not found, using defaults with environment variables")
    
    return default_config

# Load API keys from environment
def load_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables or config file"""
    api_keys = {}
    
    # Try environment first
    env_key = os.getenv("MCP_API_KEY")
    if env_key:
        api_keys[env_key] = {"name": "env_key", "created": datetime.now().isoformat()}
        logger.info(f"Using environment API key: {env_key[:8]}...")
        return api_keys
    
    # Fallback to config file
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            if "api_keys" in config:
                api_keys.update(config["api_keys"])
    except FileNotFoundError:
        pass
    
    # Generate default key if none found
    if not api_keys:
        default_key = "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
        api_keys[default_key] = {"name": "default", "created": datetime.now().isoformat()}
        logger.info(f"Using default API key: {default_key[:8]}...")
    
    return api_keys

api_keys = load_api_keys()

# User-specific MemoryOS instances
user_memory_instances: Dict[str, Memoryos] = {}

def get_memoryos_for_user(user_id: str, assistant_id: str = "mcp_assistant") -> Memoryos:
    """Get or create MemoryOS instance for specific user with proper isolation"""
    if user_id not in user_memory_instances:
        logger.info(f"Creating new MemoryOS instance for user: {user_id}")
        config = load_config()
        
        # Create MemoryOS with correct constructor parameters
        user_memory_instances[user_id] = Memoryos(
            user_id=user_id,
            openai_api_key=config.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
            data_storage_path=config.get("data_storage_path", "./memoryos_data"),
            assistant_id=assistant_id,
            openai_base_url=config.get("openai_base_url"),
            short_term_capacity=config.get("short_term_capacity", 7),
            mid_term_capacity=config.get("mid_term_capacity", 2000),
            long_term_knowledge_capacity=config.get("long_term_knowledge_capacity", 100),
            retrieval_queue_capacity=config.get("retrieval_queue_capacity", 7),
            mid_term_heat_threshold=config.get("mid_term_heat_threshold", 5.0),
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            embedding_model=config.get("embedding_model", "text-embedding-3-small")
        )
    return user_memory_instances[user_id]

# MCP 2.0 JSON-RPC Request/Response Models
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

# MCP 2.0 Tool Implementations
def handle_add_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle add_memory tool call"""
    try:
        # Extract parameters directly from MCP 2.0 arguments
        arguments = params.get("arguments", {})
        
        # Handle both nested and direct parameter formats
        if "params" in arguments:
            # MCP 2.0 client format: {"arguments": {"params": {...}}}
            actual_params = arguments["params"]
        else:
            # Direct format: {"arguments": {...}}
            actual_params = arguments
        
        user_input = actual_params.get("user_input")
        agent_response = actual_params.get("agent_response")
        user_id = actual_params.get("user_id")
        timestamp = actual_params.get("timestamp")
        meta_data = actual_params.get("meta_data", {})
        
        if not all([user_input, agent_response, user_id]):
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid parameters: user_input, agent_response, and user_id are required"
                }
            }
        
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
            meta_data=meta_data
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "message": "Memory added successfully",
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "details": {
                        "user_input_length": len(user_input),
                        "agent_response_length": len(agent_response),
                        "has_metadata": bool(meta_data)
                    }
                }, indent=2)
            }],
            "isError": False
        }
        
    except Exception as e:
        logger.error(f"Error in add_memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error adding memory: {str(e)}"
            }],
            "isError": True
        }

def handle_retrieve_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle retrieve_memory tool call"""
    try:
        # Extract parameters directly from MCP 2.0 arguments
        arguments = params.get("arguments", {})
        
        # Handle both nested and direct parameter formats
        if "params" in arguments:
            # MCP 2.0 client format: {"arguments": {"params": {...}}}
            actual_params = arguments["params"]
        else:
            # Direct format: {"arguments": {...}}
            actual_params = arguments
        
        query = actual_params.get("query")
        user_id = actual_params.get("user_id")
        relationship_with_user = actual_params.get("relationship_with_user", "assistant")
        style_hint = actual_params.get("style_hint", "")
        max_results = actual_params.get("max_results", 10)
        
        if not all([query, user_id]):
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid parameters: query and user_id are required"
                }
            }
        
        logger.info(f">>> Tool: 'retrieve_memory' called for user '{user_id}' with query: '{query[:50]}...'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Retrieve memories using MemoryOS
        result = memoryos_instance.retriever.retrieve_context(
            user_query=query,
            user_id=user_id
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "query": query,
                    "result": result
                }, indent=2)
            }],
            "isError": False
        }
        
    except Exception as e:
        logger.error(f"Error in retrieve_memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error retrieving memory: {str(e)}"
            }],
            "isError": True
        }

def handle_get_user_profile(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get_user_profile tool call"""
    try:
        # Extract parameters directly from MCP 2.0 arguments
        arguments = params.get("arguments", {})
        
        # Handle both nested and direct parameter formats
        if "params" in arguments:
            # MCP 2.0 client format: {"arguments": {"params": {...}}}
            actual_params = arguments["params"]
        else:
            # Direct format: {"arguments": {...}}
            actual_params = arguments
        
        user_id = actual_params.get("user_id")
        include_knowledge = actual_params.get("include_knowledge", True)
        include_assistant_knowledge = actual_params.get("include_assistant_knowledge", False)
        
        if not user_id:
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid parameters: user_id is required"
                }
            }
        
        logger.info(f">>> Tool: 'get_user_profile' called for user '{user_id}'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Get user profile
        profile = memoryos_instance.user_long_term_memory.get_user_profile_summary()
        
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_profile": profile
        }
        
        # Include user knowledge if requested
        if include_knowledge:
            user_knowledge = memoryos_instance.user_long_term_memory.get_user_knowledge()
            result["user_knowledge"] = user_knowledge
            result["user_knowledge_count"] = len(user_knowledge)
            
        # Include assistant knowledge if requested
        if include_assistant_knowledge:
            assistant_knowledge = memoryos_instance.assistant_long_term_memory.get_assistant_knowledge()
            result["assistant_knowledge"] = assistant_knowledge
            result["assistant_knowledge_count"] = len(assistant_knowledge)
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }],
            "isError": False
        }
        
    except Exception as e:
        logger.error(f"Error in get_user_profile: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error getting user profile: {str(e)}"
            }],
            "isError": True
        }

# MCP 2.0 Tool Registry
MCP_TOOLS = {
    "add_memory": {
        "name": "add_memory",
        "description": "Add a new memory entry to the MemoryOS system with user isolation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_input": {"type": "string", "description": "The user's input or question"},
                "agent_response": {"type": "string", "description": "The agent's response"},
                "user_id": {"type": "string", "description": "The user identifier for memory isolation"},
                "timestamp": {"type": "string", "description": "Optional timestamp in ISO format"},
                "meta_data": {"type": "object", "description": "Optional metadata dictionary"}
            },
            "required": ["user_input", "agent_response", "user_id"]
        },
        "handler": handle_add_memory
    },
    "retrieve_memory": {
        "name": "retrieve_memory",
        "description": "Retrieve relevant memories from MemoryOS system with user isolation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "user_id": {"type": "string", "description": "The user identifier for memory isolation"},
                "relationship_with_user": {"type": "string", "description": "Relationship context", "default": "assistant"},
                "style_hint": {"type": "string", "description": "Style preference", "default": ""},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10}
            },
            "required": ["query", "user_id"]
        },
        "handler": handle_retrieve_memory
    },
    "get_user_profile": {
        "name": "get_user_profile",
        "description": "Get comprehensive user profile and knowledge information with user isolation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user identifier for profile retrieval"},
                "include_knowledge": {"type": "boolean", "description": "Whether to include user knowledge entries", "default": True},
                "include_assistant_knowledge": {"type": "boolean", "description": "Whether to include assistant knowledge entries", "default": False}
            },
            "required": ["user_id"]
        },
        "handler": handle_get_user_profile
    }
}

# Create FastAPI app
app = FastAPI(title="MemoryOS Pure MCP 2.0 Remote Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication middleware
async def verify_api_key(credentials = Security(security)):
    """Verify API key from Authorization header"""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    if api_key not in api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

# Root and health check endpoints
@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "server": "MemoryOS Pure MCP 2.0 Remote Server",
        "version": "1.0.0",
        "protocol": "MCP 2.0 JSON-RPC",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "mcp": "/mcp/",
            "health": "/health"
        },
        "tools": list(MCP_TOOLS.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# MCP 2.0 Protocol Handler
@app.post("/mcp/")
async def handle_mcp_request(request: Request, api_key: str = Security(verify_api_key)):
    """Handle MCP 2.0 JSON-RPC requests"""
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
        
        logger.info(f"MCP Request: {mcp_request.method} (ID: {mcp_request.id})")
        
        # Handle initialize
        if mcp_request.method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "experimental": {},
                        "prompts": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False}
                    },
                    "serverInfo": {
                        "name": "MemoryOS Pure MCP 2.0 Server",
                        "version": "1.0.0"
                    }
                }
            })
        
        # Handle initialized notification
        elif mcp_request.method == "notifications/initialized":
            return JSONResponse({"status": "ok"}, status_code=202)
        
        # Handle tools/list
        elif mcp_request.method == "tools/list":
            tools = []
            for tool_name, tool_info in MCP_TOOLS.items():
                tools.append({
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "inputSchema": tool_info["inputSchema"]
                })
            
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "result": {"tools": tools}
            })
        
        # Handle tools/call
        elif mcp_request.method == "tools/call":
            params = mcp_request.params or {}
            tool_name = params.get("name")
            
            if tool_name not in MCP_TOOLS:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": mcp_request.id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    }
                })
            
            # Execute tool
            tool_handler = MCP_TOOLS[tool_name]["handler"]
            result = tool_handler(params)
            
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "result": result
            })
        
        # Unknown method
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": mcp_request.id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {mcp_request.method}"
                }
            })
    
    except Exception as e:
        logger.error(f"MCP request error: {e}")
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": getattr(mcp_request, 'id', None) if 'mcp_request' in locals() else None,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }, status_code=500)

async def main():
    """Run the pure MCP 2.0 server"""
    logger.info("Starting MemoryOS Pure MCP 2.0 Remote Server")
    logger.info(f"API Keys loaded: {len(api_keys)}")
    
    # Run the server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())