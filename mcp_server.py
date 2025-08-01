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
from memoryos.schema_loader import load_schema, validate_input

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

# MCP 2.0 Dual Memory Tool Implementations
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

# Dual Memory System Handlers
def handle_add_conversation_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle add_conversation_memory tool call"""
    try:
        # Extract parameters directly from MCP 2.0 arguments
        arguments = params.get("arguments", {})
        
        # Handle both nested and direct parameter formats
        if "params" in arguments:
            actual_params = arguments["params"]
        else:
            actual_params = arguments
        
        # Extract required fields
        message_id = actual_params.get("message_id")
        explanation = actual_params.get("explanation")
        user_input = actual_params.get("user_input")
        agent_response = actual_params.get("agent_response")
        user_id = actual_params.get("user_id")
        timestamp = actual_params.get("timestamp")
        meta_data = actual_params.get("meta_data", {})
        
        if not all([message_id, explanation, user_input, agent_response, user_id]):
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid parameters: message_id, explanation, user_input, agent_response, and user_id are required"
                }
            }
        
        logger.info(f">>> Tool: 'add_conversation_memory' called for user '{user_id}' with message_id '{message_id}'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Use provided timestamp or current time
        timestamp = timestamp or datetime.now().isoformat()
        
        # Add conversation memory
        result = memoryos_instance.add_conversation_memory(
            user_input=user_input,
            agent_response=agent_response,
            message_id=message_id,
            timestamp=timestamp,
            meta_data=meta_data
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": result.get("status") == "success",
                    "message": result.get("message", "Conversation memory added"),
                    "data": {
                        "status": result.get("status"),
                        "message_id": message_id,
                        "timestamp": timestamp,
                        "details": {
                            "has_meta_data": bool(meta_data),
                            "memory_processing": "Added to short-term memory, will process through memory layers"
                        }
                    }
                }, indent=2)
            }],
            "isError": result.get("status") != "success"
        }
        
    except Exception as e:
        logger.error(f"Error in add_conversation_memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": False,
                    "message": f"Error adding conversation memory: {str(e)}",
                    "data": {
                        "status": "error",
                        "message_id": actual_params.get("message_id", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }
                })
            }],
            "isError": True
        }

def handle_retrieve_conversation_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle retrieve_conversation_memory tool call"""
    try:
        # Extract parameters
        arguments = params.get("arguments", {})
        if "params" in arguments:
            actual_params = arguments["params"]
        else:
            actual_params = arguments
        
        # Extract fields
        explanation = actual_params.get("explanation")
        query = actual_params.get("query")
        user_id = actual_params.get("user_id")
        message_id = actual_params.get("message_id")
        time_range = actual_params.get("time_range")
        max_results = actual_params.get("max_results", 10)
        
        if not all([explanation, query, user_id]):
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid params: explanation, query, and user_id are required"
                }
            }
        
        logger.info(f">>> Tool: 'retrieve_conversation_memory' called for user '{user_id}' with query: '{query[:50]}...'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Retrieve memories using MemoryOS
        result = memoryos_instance.retriever.retrieve_context(
            user_query=query,
            user_id=user_id
        )
        
        # Format for conversation memory schema
        conversations = []
        for entry in result.get("short_term_memory", [])[:max_results]:
            conversations.append({
                "message_id": entry.get("message_id", "unknown"),
                "conversation_timestamp": entry.get("timestamp", ""),
                "user_input": entry.get("user_input", ""),
                "agent_response": entry.get("agent_response", ""),
                "meta_data": entry.get("meta_data", {}),
                "has_execution_memory": False,  # TODO: Check if execution exists
                "relevance_score": entry.get("similarity_score", 0.5)
            })
        
        # Add mid-term entries
        for entry in result.get("retrieved_pages", []):
            conversations.append({
                "message_id": entry.get("meta_info", {}).get("segment_id", "unknown"),
                "conversation_timestamp": entry.get("timestamp", ""),
                "user_input": entry.get("user_input", ""),
                "agent_response": entry.get("agent_response", ""),
                "meta_data": entry.get("meta_info", {}),
                "has_execution_memory": False,  # TODO: Check if execution exists
                "relevance_score": entry.get("meta_info", {}).get("similarity_score", 0.5)
            })
        
        # Determine query type
        query_type = "specific_message" if message_id else "general"
        if time_range:
            query_type = "time_filtered"
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "message": f"Retrieved {len(conversations)} conversation memories",
                    "data": {
                        "status": "success",
                        "query": query,
                        "explanation": explanation,
                        "query_type": query_type,
                        "requested_message_id": message_id,
                        "retrieval_timestamp": datetime.now().isoformat(),
                        "time_range": time_range,
                        "conversations": conversations[:max_results],
                        "total_found": len(conversations),
                        "returned_count": min(len(conversations), max_results),
                        "max_results_applied": len(conversations) > max_results
                    }
                }, indent=2)
            }],
            "isError": False
        }
        
    except Exception as e:
        logger.error(f"Error in retrieve_conversation_memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": False,
                    "message": f"Error retrieving conversation memory: {str(e)}",
                    "data": {
                        "status": "error",
                        "query": actual_params.get("query", ""),
                        "explanation": actual_params.get("explanation", ""),
                        "query_type": "error",
                        "retrieval_timestamp": datetime.now().isoformat(),
                        "conversations": [],
                        "total_found": 0,
                        "returned_count": 0,
                        "max_results_applied": False
                    }
                })
            }],
            "isError": True
        }

def handle_add_execution_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle add_execution_memory tool call"""
    try:
        # Extract parameters
        arguments = params.get("arguments", {})
        if "params" in arguments:
            actual_params = arguments["params"]
        else:
            actual_params = arguments
        
        # Extract required fields (flattened structure)
        message_id = actual_params.get("message_id")
        explanation = actual_params.get("explanation")
        execution_summary = actual_params.get("execution_summary")
        tools_used = actual_params.get("tools_used", [])
        errors = actual_params.get("errors", [])
        observations = actual_params.get("observations")
        success = actual_params.get("success")
        user_id = actual_params.get("user_id")
        duration_ms = actual_params.get("duration_ms")
        timestamp = actual_params.get("timestamp")
        meta_data = actual_params.get("meta_data", {})
        
        if not all([message_id, explanation, execution_summary, user_id, observations]) or success is None:
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid parameters: message_id, explanation, execution_summary, observations, success, and user_id are required"
                }
            }
        
        logger.info(f">>> Tool: 'add_execution_memory' called for user '{user_id}' with message_id '{message_id}'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Use provided timestamp or current time
        timestamp = timestamp or datetime.now().isoformat()
        
        # Add execution memory
        result = memoryos_instance.add_execution_memory(
            message_id=message_id,
            execution_summary=execution_summary,
            tools_used=tools_used,
            errors=errors,
            observations=observations,
            success=success,
            duration_ms=duration_ms,
            timestamp=timestamp,
            meta_data=meta_data
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": result.get("status") == "success",
                    "message": result.get("message", "Execution memory added"),
                    "data": {
                        "status": result.get("status"),
                        "message_id": message_id,
                        "timestamp": timestamp,
                        "details": {
                            "execution_summary": execution_summary[:100] + "..." if len(execution_summary) > 100 else execution_summary,
                            "tools_used": tools_used,
                            "errors": errors,
                            "duration_ms": duration_ms,
                            "success": success,
                            "has_meta_data": bool(meta_data),
                            "memory_processing": "Added to execution memory with embedding generation"
                        }
                    }
                }, indent=2)
            }],
            "isError": result.get("status") != "success"
        }
        
    except Exception as e:
        logger.error(f"Error in add_execution_memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": False,
                    "message": f"Error adding execution memory: {str(e)}",
                    "data": {
                        "status": "error",
                        "message_id": actual_params.get("message_id", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }
                })
            }],
            "isError": True
        }

def handle_retrieve_execution_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle retrieve_execution_memory tool call"""
    try:
        # Extract parameters
        arguments = params.get("arguments", {})
        if "params" in arguments:
            actual_params = arguments["params"]
        else:
            actual_params = arguments
        
        # Extract fields
        explanation = actual_params.get("explanation")
        query = actual_params.get("query")
        user_id = actual_params.get("user_id")
        message_id = actual_params.get("message_id")
        max_results = actual_params.get("max_results", 10)
        
        if not all([explanation, query, user_id]):
            return {
                "error": {
                    "code": -32602,
                    "message": "Invalid params: explanation, query, and user_id are required"
                }
            }
        
        logger.info(f">>> Tool: 'retrieve_execution_memory' called for user '{user_id}' with query: '{query[:50]}...'")
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id)
        
        # Retrieve execution memories
        result = memoryos_instance.retrieve_execution_memory(
            query=query,
            message_id=message_id,
            max_results=max_results
        )
        
        # Format for execution memory schema
        executions = []
        for exec_record in result.get("results", []):
            executions.append({
                "message_id": exec_record.get("message_id", "unknown"),
                "execution_timestamp": exec_record.get("timestamp", ""),
                "execution_details": {
                    "execution_summary": exec_record.get("execution_summary", ""),
                    "tools_used": exec_record.get("tools_used", []),
                    "errors": exec_record.get("errors", []),
                    "observations": exec_record.get("observations", "")
                },
                "success": exec_record.get("success", False),
                "duration_ms": exec_record.get("duration_ms", 0),
                "relevance_score": exec_record.get("similarity_score", 0.5)
            })
        
        # Determine query type
        query_type = "specific_message" if message_id else "general"
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "message": f"Retrieved {len(executions)} execution memories",
                    "data": {
                        "status": "success",
                        "query": query,
                        "explanation": explanation,
                        "query_type": query_type,
                        "requested_message_id": message_id,
                        "retrieval_timestamp": datetime.now().isoformat(),
                        "executions": executions[:max_results],
                        "total_found": len(executions),
                        "returned_count": min(len(executions), max_results),
                        "max_results_applied": len(executions) > max_results
                    }
                }, indent=2)
            }],
            "isError": False
        }
        
    except Exception as e:
        logger.error(f"Error in retrieve_execution_memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": False,
                    "message": f"Error retrieving execution memory: {str(e)}",
                    "data": {
                        "status": "error",
                        "query": actual_params.get("query", ""),
                        "explanation": actual_params.get("explanation", ""),
                        "query_type": "error",
                        "retrieval_timestamp": datetime.now().isoformat(),
                        "executions": [],
                        "total_found": 0,
                        "returned_count": 0,
                        "max_results_applied": False
                    }
                })
            }],
            "isError": True
        }

# MCP 2.0 Tool Registry - Dual Memory System
MCP_TOOLS = {
    # Legacy tools (backward compatibility)
    "add_memory": {
        "name": "add_memory",
        "description": "Add a new memory entry to the MemoryOS system with user isolation (legacy - use add_conversation_memory instead)",
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
        "handler": handle_add_conversation_memory  # Reuse new handler
    },
    "retrieve_memory": {
        "name": "retrieve_memory",
        "description": "Retrieve relevant memories from MemoryOS system with user isolation (legacy - use retrieve_conversation_memory instead)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "user_id": {"type": "string", "description": "The user identifier for memory isolation"},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10}
            },
            "required": ["query", "user_id"]
        },
        "handler": handle_retrieve_conversation_memory  # Reuse new handler
    },
    
    # Core dual memory system tools
    "add_conversation_memory": {
        "name": "add_conversation_memory",
        "description": "Store conversation pair (user input and agent response) in MemoryOS dual memory system",
        "inputSchema": load_schema("add_conversation_input.json"),
        "handler": handle_add_conversation_memory
    },
    "retrieve_conversation_memory": {
        "name": "retrieve_conversation_memory", 
        "description": "Retrieve conversation pairs with execution links from MemoryOS dual memory system",
        "inputSchema": load_schema("retrieve_conversation_input.json"),
        "handler": handle_retrieve_conversation_memory
    },
    "add_execution_memory": {
        "name": "add_execution_memory",
        "description": "Store execution details linked to conversation memory via message_id",
        "inputSchema": load_schema("add_execution_input.json"),
        "handler": handle_add_execution_memory
    },
    "retrieve_execution_memory": {
        "name": "retrieve_execution_memory",
        "description": "Retrieve execution patterns and details for learning from past problem-solving approaches",
        "inputSchema": load_schema("retrieve_execution_input.json"),
        "handler": handle_retrieve_execution_memory
    },
    
    # Utility tools
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