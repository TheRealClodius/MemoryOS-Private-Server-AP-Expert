#!/usr/bin/env python3
"""
MemoryOS MCP Server using Anthropic MCP SDK 1.2.0+ with FastMCP framework
Optimized for Gemini client integration with OpenAI embeddings and FAISS-GPU support
"""

import sys
import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import time
import hashlib
import secrets
from collections import defaultdict

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool
    from pydantic import BaseModel, Field
    from mcp.server.models import InitializationOptions
    from mcp.server.session import ServerSession
    from mcp.server import Server
    from mcp.shared.exceptions import McpError
    from mcp.types import (
    JSONRPCRequest, 
    JSONRPCResponse, 
    JSONRPCError,
    InitializeRequest,
    InitializeResult,
    ServerCapabilities,
        CallToolRequest,
        CallToolResult,
        ListToolsRequest,
        ListToolsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListPromptsRequest,
        ListPromptsResult,
        ReadResourceRequest,
        ReadResourceResult,
        GetPromptRequest,
        GetPromptResult,
        TextContent,
        ImageContent,
        EmbeddedResource
    )
except ImportError as e:
    print(f"ERROR: Failed to import MCP SDK. Please install: pip install mcp", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

# Import FastAPI for HTTP health check endpoint
try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends, Security
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response as StarletteResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available, HTTP health check disabled", file=sys.stderr)

try:
    from memoryos import Memoryos
except ImportError as e:
    print(f"ERROR: Failed to import MemoryOS. Ensure memoryos package is available.", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

# Pydantic models for structured tool returns per 2025 MCP specifications
class MemoryOperationResult(BaseModel):
    """Structured result for memory operations"""
    status: str = Field(description="Operation status: 'success' or 'error'")
    message: str = Field(description="Human-readable status message")
    timestamp: str = Field(description="Operation timestamp in ISO format")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional operation details")

class MemoryEntry(BaseModel):
    """Individual memory entry structure"""
    user_input: str = Field(description="User's input or question")
    agent_response: str = Field(description="Assistant's response")
    timestamp: str = Field(description="When this interaction occurred")
    meta_info: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class KnowledgeEntry(BaseModel):
    """Knowledge base entry structure"""
    knowledge: str = Field(description="Knowledge content")
    timestamp: str = Field(description="When this knowledge was added")
    source: Optional[str] = Field(default=None, description="Source of the knowledge")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    similarity_score: Optional[float] = Field(default=None, description="Similarity to query")

class MemoryRetrievalResult(BaseModel):
    """Structured result for memory retrieval"""
    status: str = Field(description="Operation status")
    query: str = Field(description="Original query")
    timestamp: str = Field(description="Retrieval timestamp")
    user_profile: str = Field(description="User profile summary")
    short_term_memory: List[MemoryEntry] = Field(description="Recent conversation history")
    short_term_count: int = Field(description="Number of short-term memories")
    retrieved_pages: List[MemoryEntry] = Field(description="Relevant past interactions")
    retrieved_user_knowledge: List[KnowledgeEntry] = Field(description="Relevant user knowledge")
    retrieved_assistant_knowledge: List[KnowledgeEntry] = Field(description="Relevant assistant knowledge")

class UserProfileResult(BaseModel):
    """Structured result for user profile"""
    status: str = Field(description="Operation status")
    timestamp: str = Field(description="Profile timestamp")
    user_id: str = Field(description="User identifier")
    assistant_id: str = Field(description="Assistant identifier")
    user_profile: str = Field(description="User profile summary")
    user_knowledge: Optional[List[KnowledgeEntry]] = Field(default=None, description="User knowledge entries")
    user_knowledge_count: Optional[int] = Field(default=None, description="Number of user knowledge entries")
    assistant_knowledge: Optional[List[KnowledgeEntry]] = Field(default=None, description="Assistant knowledge entries")
    assistant_knowledge_count: Optional[int] = Field(default=None, description="Number of assistant knowledge entries")

# Global MemoryOS instance
memoryos_instance: Optional[Memoryos] = None

# User management - add proper user isolation
_user_instances: Dict[str, Memoryos] = {}
_session_user_map: Dict[str, str] = {}

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
            openai_api_key=config["openai_api_key"],
            openai_base_url=config["openai_base_url"],
            data_storage_path=config["data_storage_path"],
            short_term_capacity=config["short_term_capacity"],
            mid_term_capacity=config["mid_term_capacity"],
            long_term_knowledge_capacity=config["long_term_knowledge_capacity"],
            retrieval_queue_capacity=config["retrieval_queue_capacity"],
            mid_term_heat_threshold=config["mid_term_heat_threshold"],
            llm_model=config["llm_model"],
            embedding_model=config["embedding_model"]
        )
        
        # Cache the instance
        _user_instances[cache_key] = instance
        print(f"🔐 Created isolated MemoryOS instance for user: {user_id}", file=sys.stderr)
        return instance
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MemoryOS for user {user_id}: {str(e)}")

def get_user_id_from_session(session_id: str) -> str:
    """Get user ID from session, with fallback to API key mapping"""
    if session_id in _session_user_map:
        return _session_user_map[session_id]
    
    # If no explicit user mapping, use session ID as user ID
    # This ensures isolation even without explicit user setup
    user_id = f"session_{session_id[:8]}"
    _session_user_map[session_id] = user_id
    return user_id

def set_user_for_session(session_id: str, user_id: str):
    """Set user ID for a session"""
    _session_user_map[session_id] = user_id
    print(f"🔐 Mapped session {session_id[:8]}... to user {user_id}", file=sys.stderr)

# Security Configuration
class SecurityConfig:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
        self.enable_cors = os.getenv("ENABLE_CORS", "true").lower() == "true"
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        self.trusted_hosts = os.getenv("TRUSTED_HOSTS", "*").split(",")
        self.require_https = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"
    
    def _load_api_keys(self) -> Dict[str, Dict[str, str]]:
        """Load API keys from environment or generate default"""
        api_keys = {}
        
        # Load from environment
        env_keys = os.getenv("MCP_API_KEYS", "")
        if env_keys:
            for key_pair in env_keys.split(","):
                if ":" in key_pair:
                    name, key = key_pair.split(":", 1)
                    api_keys[key] = {"name": name.strip(), "created": datetime.now().isoformat()}
        
        # Generate default key if none provided
        if not api_keys:
            default_key = os.getenv("MCP_API_KEY")
            if not default_key:
                default_key = secrets.token_urlsafe(32)
                print(f"🔑 Generated API Key: {default_key}", file=sys.stderr)
                print("🔑 Set MCP_API_KEY environment variable to use this key", file=sys.stderr)
            
            api_keys[default_key] = {"name": "default", "created": datetime.now().isoformat()}
        
        return api_keys
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, str]]:
        """Validate API key and return key info"""
        return self.api_keys.get(api_key)

# Security configuration will be initialized in init_server()
security_config = None

# Rate limiting storage
rate_limit_storage = defaultdict(list)

# Security Middleware Classes
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Server"] = "MemoryOS-MCP"
        
        # HTTPS enforcement
        if security_config.require_https and request.url.scheme != "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/", "/health"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        
        # Check rate limit
        now = time.time()
        window_start = now - security_config.rate_limit_window
        
        # Clean old requests
        rate_limit_storage[client_ip] = [
            req_time for req_time in rate_limit_storage[client_ip] 
            if req_time > window_start
        ]
        
        # Check if over limit
        if len(rate_limit_storage[client_ip]) >= security_config.rate_limit_requests:
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                headers={"Retry-After": str(security_config.rate_limit_window)}
            )
        
        # Add current request
        rate_limit_storage[client_ip].append(now)
        
        return await call_next(request)

# Authentication schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def get_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> str:
    """Extract API key from header or bearer token"""
    global security_config
    
    # Ensure security config is initialized
    if security_config is None:
        security_config = SecurityConfig()
    
    # Try header first
    if api_key_header:
        key_info = security_config.validate_api_key(api_key_header)
        if key_info:
            return api_key_header
    
    # Try bearer token
    if bearer_token:
        key_info = security_config.validate_api_key(bearer_token.credentials)
        if key_info:
            return bearer_token.credentials
    
    # Check if authentication is disabled for development
    if os.getenv("DISABLE_AUTH", "false").lower() == "true":
        print("⚠️  Authentication disabled - development mode only!", file=sys.stderr)
        return "dev-mode"
    
    # Debug logging for failed authentication
    print(f"❌ API key validation failed. Received header: {api_key_header[:8] if api_key_header else 'None'}...", file=sys.stderr)
    print(f"❌ Available keys: {[key[:8] + '...' for key in security_config.api_keys.keys()]}", file=sys.stderr)
    
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"}
    )

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with environment variable fallbacks"""
    config = {}
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}", file=sys.stderr)
    
    # Environment variable fallbacks with defaults
    # Dynamic user ID generation if not provided
    if not config.get("user_id") and not os.getenv("MEMORYOS_USER_ID"):
        import uuid
        config["user_id"] = f"user_{str(uuid.uuid4())[:8]}"
    else:
        config.setdefault("user_id", os.getenv("MEMORYOS_USER_ID", "default_user"))
    
    config.setdefault("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
    config.setdefault("openai_base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    
    # User-specific data storage path
    base_path = os.getenv("MEMORYOS_DATA_PATH", "./memoryos_data")
    config.setdefault("data_storage_path", f"{base_path}/{config['user_id']}")
    
    config.setdefault("assistant_id", os.getenv("MEMORYOS_ASSISTANT_ID", "mcp_assistant"))
    config.setdefault("llm_model", os.getenv("MEMORYOS_LLM_MODEL", "gpt-4o-mini"))
    config.setdefault("embedding_model", os.getenv("MEMORYOS_EMBEDDING_MODEL", "text-embedding-3-small"))
    config.setdefault("short_term_capacity", int(os.getenv("MEMORYOS_SHORT_TERM_CAPACITY", "10")))
    config.setdefault("mid_term_capacity", int(os.getenv("MEMORYOS_MID_TERM_CAPACITY", "2000")))
    config.setdefault("long_term_knowledge_capacity", int(os.getenv("MEMORYOS_KNOWLEDGE_CAPACITY", "100")))
    config.setdefault("retrieval_queue_capacity", int(os.getenv("MEMORYOS_RETRIEVAL_CAPACITY", "7")))
    config.setdefault("mid_term_heat_threshold", float(os.getenv("MEMORYOS_HEAT_THRESHOLD", "5.0")))
    
    # Validate required fields
    if not config.get("openai_api_key"):
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or add to config.json")
    
    return config

def init_memoryos(config: Dict[str, Any]) -> Memoryos:
    """Initialize MemoryOS instance with configuration"""
    try:
        return Memoryos(
            user_id=config["user_id"],
            openai_api_key=config["openai_api_key"],
            openai_base_url=config["openai_base_url"],
            data_storage_path=config["data_storage_path"],
            assistant_id=config["assistant_id"],
            short_term_capacity=config["short_term_capacity"],
            mid_term_capacity=config["mid_term_capacity"],
            long_term_knowledge_capacity=config["long_term_knowledge_capacity"],
            retrieval_queue_capacity=config["retrieval_queue_capacity"],
            mid_term_heat_threshold=config["mid_term_heat_threshold"],
            llm_model=config["llm_model"],
            embedding_model=config["embedding_model"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MemoryOS: {str(e)}")

# Create FastMCP server instance
mcp = FastMCP("MemoryOS")

# Create FastAPI app with security
app = FastAPI(
    title="MemoryOS MCP Server (Secure)",
    version="1.0.0",
    description="Remote MCP server for MemoryOS with authentication",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

# CORS middleware will be added after security config is initialized in init_server()

# Trusted host middleware will be added after security config is initialized in init_server()

# StreamableHTTP MCP Server Implementation
class StreamableHTTPMCPServer:
    def __init__(self):
        self.server = Server("MemoryOS")
        self.sessions: Dict[str, ServerSession] = {}
        self.current_session_id: Optional[str] = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="add_memory",
                    description="Add a new memory to the MemoryOS system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_input": {"type": "string", "description": "The user's input or question"},
                            "agent_response": {"type": "string", "description": "The agent's response"},
                            "user_id": {"type": "string", "description": "The user identifier for memory isolation (required)"},
                            "timestamp": {"type": "string", "description": "Optional timestamp in ISO format"},
                            "meta_data": {"type": "object", "description": "Optional metadata dictionary"}
                        },
                        "required": ["user_input", "agent_response", "user_id"]
                    }
                ),
                Tool(
                    name="retrieve_memory",
                    description="Retrieve relevant memories from MemoryOS",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "user_id": {"type": "string", "description": "The user identifier for memory isolation (required)"},
                            "relationship_with_user": {"type": "string", "description": "Relationship context (assistant, friend, colleague, etc.)", "default": "assistant"},
                            "style_hint": {"type": "string", "description": "Style preference for response formatting", "default": ""},
                            "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results"}
                        },
                        "required": ["query", "user_id"]
                    }
                ),
                Tool(
                    name="get_user_profile",
                    description="Get comprehensive user profile information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "The user identifier for profile retrieval (required)"},
                            "include_knowledge": {"type": "boolean", "description": "Whether to include user knowledge entries", "default": True},
                            "include_assistant_knowledge": {"type": "boolean", "description": "Whether to include assistant knowledge entries", "default": False}
                        },
                        "required": ["user_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if name == "add_memory":
                    result = await add_memory(
                        user_input=arguments.get("user_input", ""),
                        agent_response=arguments.get("agent_response", ""),
                        user_id=arguments.get("user_id", ""),
                        timestamp=arguments.get("timestamp"),
                        meta_data=arguments.get("meta_data")
                    )
                    return [TextContent(type="text", text=json.dumps(result.dict()))]
                
                elif name == "retrieve_memory":
                    result = await retrieve_memory(
                        query=arguments.get("query", ""),
                        user_id=arguments.get("user_id", ""),
                        relationship_with_user=arguments.get("relationship_with_user", "assistant"),
                        style_hint=arguments.get("style_hint", ""),
                        max_results=arguments.get("max_results", 10)
                    )
                    return [TextContent(type="text", text=json.dumps(result.dict()))]
                
                elif name == "get_user_profile":
                    result = await get_user_profile(
                        user_id=arguments.get("user_id", ""),
                        include_knowledge=arguments.get("include_knowledge", True),
                        include_assistant_knowledge=arguments.get("include_assistant_knowledge", False)
                    )
                    return [TextContent(type="text", text=json.dumps(result.dict()))]
                
                else:
                    raise McpError(f"Unknown tool: {name}")
            
            except Exception as e:
                print(f"Error in call_tool: {e}", file=sys.stderr)
                raise McpError(f"Tool execution failed: {str(e)}")

# Handler functions removed - now using direct tool functions with user_id parameter

# Initialize StreamableHTTP server
streamable_server = StreamableHTTPMCPServer()

# Update MCP endpoints with authentication
@app.post("/mcp")
async def handle_mcp_request(
    request: Request, 
    api_key: str = Depends(get_api_key)
) -> JSONResponse:
    """Handle authenticated MCP JSON-RPC requests via StreamableHTTP"""
    try:
        # Log authenticated request
        key_info = security_config.validate_api_key(api_key)
        if key_info:
            print(f"🔐 Authenticated request from: {key_info['name']}", file=sys.stderr)
        
        # Parse JSON-RPC request
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        json_request = json.loads(body.decode())
        
        # Map API key to user ID for isolation
        user_id = request.headers.get("X-User-ID")
        if not user_id:
            # Use API key name as default user ID for isolation
            user_id = key_info.get("name", f"user_{api_key[:8]}") if key_info else f"user_{api_key[:8]}"
        
        print(f"🔐 Mapped session {api_key[:8]}... to user {user_id}", file=sys.stderr)
        
        # Handle MCP request directly
        method = json_request.get("method")
        params = json_request.get("params", {})
        request_id = json_request.get("id")
        
        if method == "initialize":
            # Return initialization response
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "logging": {},
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "MemoryOS",
                        "version": "1.0.0"
                    }
                }
            }
        elif method == "tools/list":
            # Return available tools
            tools = [
                {
                    "name": "add_memory",
                    "description": "Add a new memory to the MemoryOS system",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_input": {"type": "string", "description": "The user's input or question"},
                            "agent_response": {"type": "string", "description": "The agent's response"},
                            "user_id": {"type": "string", "description": "The user identifier for memory isolation (required)"},
                            "timestamp": {"type": "string", "description": "Optional timestamp in ISO format"},
                            "meta_data": {"type": "object", "description": "Optional metadata dictionary"}
                        },
                        "required": ["user_input", "agent_response", "user_id"]
                    }
                },
                {
                    "name": "retrieve_memory",
                    "description": "Retrieve relevant memories from MemoryOS",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "user_id": {"type": "string", "description": "The user identifier for memory isolation (required)"},
                            "relationship_with_user": {"type": "string", "description": "Relationship context", "default": "assistant"},
                            "style_hint": {"type": "string", "description": "Style preference", "default": ""},
                            "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results"}
                        },
                        "required": ["query", "user_id"]
                    }
                },
                {
                    "name": "get_user_profile",
                    "description": "Get comprehensive user profile information",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "The user identifier for profile retrieval (required)"},
                            "include_knowledge": {"type": "boolean", "description": "Whether to include user knowledge entries", "default": True},
                            "include_assistant_knowledge": {"type": "boolean", "description": "Whether to include assistant knowledge entries", "default": False}
                        },
                        "required": ["user_id"]
                    }
                }
            ]
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": tools
                }
            }
        elif method == "tools/call":
            # Handle tool calls
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "add_memory":
                result = await add_memory(
                    user_input=arguments.get("user_input", ""),
                    agent_response=arguments.get("agent_response", ""),
                    user_id=arguments.get("user_id", user_id),
                    timestamp=arguments.get("timestamp"),
                    meta_data=arguments.get("meta_data")
                )
            elif tool_name == "retrieve_memory":
                result = await retrieve_memory(
                    query=arguments.get("query", ""),
                    user_id=arguments.get("user_id", user_id),
                    relationship_with_user=arguments.get("relationship_with_user", "assistant"),
                    style_hint=arguments.get("style_hint", ""),
                    max_results=arguments.get("max_results", 10)
                )
            elif tool_name == "get_user_profile":
                result = await get_user_profile(
                    user_id=arguments.get("user_id", user_id),
                    include_knowledge=arguments.get("include_knowledge", True),
                    include_assistant_knowledge=arguments.get("include_assistant_knowledge", False)
                )
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
                return JSONResponse(content=response, status_code=400)
            
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result.model_dump(), indent=2)
                        }
                    ]
                }
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }
        
        return JSONResponse(content=response)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        print(f"Error processing authenticated MCP request: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp")
async def handle_mcp_sse(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Handle authenticated MCP Server-Sent Events (SSE) endpoint"""
    # SSE implementation would go here
    raise HTTPException(status_code=501, detail="SSE not implemented yet")

@app.delete("/mcp")
async def handle_mcp_disconnect(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Handle authenticated MCP session termination"""
    session_id = request.headers.get("X-Session-ID", f"{api_key[:8]}-default")
    if session_id in streamable_server.sessions:
        del streamable_server.sessions[session_id]
        print(f"🔐 Session terminated for API key: {api_key[:8]}...", file=sys.stderr)
    return {"status": "disconnected"}

# Public endpoints (no authentication required)
@app.get("/")
async def health_check():
    """Public health check endpoint"""
    try:
        return JSONResponse({
            "status": "healthy",
            "service": "MemoryOS MCP Server (Secure)",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "authentication": "enabled",
            "capabilities": [
                "add_memory",
                "retrieve_memory", 
                "get_user_profile"
            ]
        })
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
    """Public detailed health check"""
    try:
        # Don't initialize MemoryOS for health check to avoid dependencies
        return JSONResponse({
            "status": "healthy",
            "service": "MemoryOS MCP Server (Secure)",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "authentication": {
                "enabled": True,
                "api_keys_configured": len(security_config.api_keys),
                "rate_limiting": {
                    "requests_per_window": security_config.rate_limit_requests,
                    "window_seconds": security_config.rate_limit_window
                }
            },
            "security": {
                "cors_enabled": security_config.enable_cors,
                "https_required": security_config.require_https,
                "trusted_hosts": security_config.trusted_hosts
            },
            "active_sessions": len(streamable_server.sessions)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Admin endpoints (require authentication)
@app.get("/admin/sessions")
async def list_sessions(api_key: str = Depends(get_api_key)):
    """List active MCP sessions"""
    return {
        "active_sessions": len(streamable_server.sessions),
        "sessions": list(streamable_server.sessions.keys())
    }

@app.get("/admin/stats")
async def get_stats(api_key: str = Depends(get_api_key)):
    """Get detailed server statistics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "sessions": len(streamable_server.sessions),
        "api_keys": len(security_config.api_keys),
        "rate_limit_stats": {
            "active_clients": len(rate_limit_storage),
            "total_requests": sum(len(reqs) for reqs in rate_limit_storage.values())
        }
    }

async def init_server():
    """Initialize the MemoryOS server (no global instance needed)"""
    global security_config
    
    # Initialize security configuration after environment variables are set
    if security_config is None:
        security_config = SecurityConfig()
        print(f"🔑 Security initialized with {len(security_config.api_keys)} API key(s)", file=sys.stderr)
        for key, info in security_config.api_keys.items():
            print(f"🔑 Loaded API key: {key[:8]}... (name: {info['name']})", file=sys.stderr)
        
        # Add CORS middleware now that security config is initialized
        if security_config.enable_cors:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=security_config.allowed_origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "DELETE"],
                allow_headers=["*"],
            )
            print("🌐 CORS middleware enabled", file=sys.stderr)
        
        # Add trusted host middleware now that security config is initialized
        if security_config.trusted_hosts != ["*"]:
            from fastapi.middleware.trustedhost import TrustedHostMiddleware
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=security_config.trusted_hosts
            )
            print("🔒 Trusted host middleware enabled", file=sys.stderr)
    
    # Load configuration to verify setup
    print("Loading MemoryOS configuration...", file=sys.stderr)
    config = load_config()
    
    print(f"🔐 MemoryOS MCP Server started with USER ISOLATION", file=sys.stderr)
    print(f"🔐 Each API key/session gets isolated memory instances", file=sys.stderr)
    print(f"Data storage: {config['data_storage_path']}", file=sys.stderr)
    print(f"LLM model: {config['llm_model']}, Embedding model: {config['embedding_model']}", file=sys.stderr)
    
    return config

async def initialize_memoryos():
    """Initialize MemoryOS instance for MCP server"""
    await init_server()

async def run_streamable_http_server():
    """Run the StreamableHTTP MCP server"""
    try:
        # Initialize MemoryOS
        print("Initializing MemoryOS...", file=sys.stderr)
        await initialize_memoryos()
        
        print("MemoryOS MCP Server starting on StreamableHTTP transport...", file=sys.stderr)
        
        # Run server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", "3000")),
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        print(f"Error running StreamableHTTP server: {e}", file=sys.stderr)
        raise

async def main():
    """Main entry point - now defaults to StreamableHTTP"""
    mode = os.getenv("SERVER_MODE", "streamable-http").lower()
    
    if mode == "stdio":
        # Legacy stdio mode for backward compatibility
        print("Running in legacy stdio mode", file=sys.stderr)
        await mcp.run(transport="stdio")
    elif mode == "streamable-http" or mode == "http":
        # Modern StreamableHTTP mode (default)
        await run_streamable_http_server()
    else:
        print(f"Unknown server mode: {mode}", file=sys.stderr)
        print("Available modes: stdio, streamable-http, http", file=sys.stderr)
        sys.exit(1)

# Add MCP tool functions back
@mcp.tool()
async def add_memory(
    user_input: str,
    agent_response: str,
    user_id: str,
    timestamp: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None
) -> MemoryOperationResult:
    """
    Add a new memory entry to MemoryOS system with user isolation.
    
    Stores conversation pairs (user input + agent response) in the hierarchical memory system
    for building persistent dialogue history and contextual understanding.
    
    Args:
        user_input: The user's input, question, or statement
        agent_response: The assistant's response to the user input
        user_id: The user identifier for memory isolation (required)
        timestamp: Optional timestamp in ISO format (uses current time if not provided)
        meta_data: Optional metadata dictionary for additional context
    
    Returns:
        MemoryOperationResult with operation status and details
    """
    try:
        # Validate user_id
        if not user_id or not user_id.strip():
            return MemoryOperationResult(
                status="error",
                message="user_id is required and cannot be empty",
                timestamp=datetime.now().isoformat()
            )
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id.strip())
        
        # Validate inputs
        if not user_input or not user_input.strip():
            return MemoryOperationResult(
                status="error",
                message="user_input cannot be empty",
                timestamp=datetime.now().isoformat()
            )
        
        if not agent_response or not agent_response.strip():
            return MemoryOperationResult(
                status="error", 
                message="agent_response cannot be empty",
                timestamp=datetime.now().isoformat()
            )
        
        # Add memory to user's specific MemoryOS instance
        result = memoryos_instance.add_memory(
            user_input=user_input.strip(),
            agent_response=agent_response.strip(),
            timestamp=timestamp,
            meta_data=meta_data
        )
        
        if result.get("status") == "success":
            return MemoryOperationResult(
                status="success",
                message=f"Memory successfully added to isolated MemoryOS for user {user_id}",
                timestamp=datetime.now().isoformat(),
                details={
                    "user_id": user_id,
                    "user_input_length": len(user_input),
                    "agent_response_length": len(agent_response),
                    "has_meta_data": meta_data is not None,
                    "memory_layers_updated": ["short_term"]
                }
            )
        else:
            return MemoryOperationResult(
                status="error",
                message=result.get("message", "Unknown error occurred"),
                timestamp=datetime.now().isoformat()
            )
    
    except Exception as e:
        return MemoryOperationResult(
            status="error",
            message=f"Error adding memory: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@mcp.tool()
async def retrieve_memory(
    query: str,
    user_id: str,
    relationship_with_user: str = "assistant",
    style_hint: str = "",
    max_results: int = 10
) -> MemoryRetrievalResult:
    """
    Retrieve relevant memories and context from MemoryOS system with user isolation.
    
    Searches across all memory layers (short-term, mid-term, long-term) using semantic similarity
    to find relevant conversation history, user knowledge, and assistant knowledge.
    
    Args:
        query: The search query or topic to find relevant memories for
        user_id: The user identifier for memory isolation (required)
        relationship_with_user: Relationship context (assistant, friend, colleague, etc.)
        style_hint: Style preference for response formatting
        max_results: Maximum number of results to return per category (default: 10)
    
    Returns:
        MemoryRetrievalResult with comprehensive memory context
    """
    try:
        # Validate user_id
        if not user_id or not user_id.strip():
            return MemoryRetrievalResult(
                status="error",
                query=query,
                timestamp=datetime.now().isoformat(),
                user_profile="user_id is required and cannot be empty",
                short_term_memory=[],
                short_term_count=0,
                retrieved_pages=[],
                retrieved_user_knowledge=[],
                retrieved_assistant_knowledge=[]
            )
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id.strip())
        
        # Validate query
        if not query or not query.strip():
            return MemoryRetrievalResult(
                status="error",
                query=query,
                timestamp=datetime.now().isoformat(),
                user_profile="",
                short_term_memory=[],
                short_term_count=0,
                retrieved_pages=[],
                retrieved_user_knowledge=[],
                retrieved_assistant_knowledge=[]
            )
        
        # Generate embedding for the query (fallback to None if generation fails)
        query_embedding = None
        try:
            query_embedding = memoryos_instance._generate_embedding(query.strip())
        except Exception as e:
            print(f"Warning: Embedding generation failed, using fallback: {e}", file=sys.stderr)
            # Continue with None embedding - short-term memory will still work
        
        # Retrieve context from all memory layers
        retrieval_results = memoryos_instance.retriever.retrieve_context(
            user_query=query.strip(),
            user_id=memoryos_instance.user_id,
            query_embedding=query_embedding
        )
        
        # Get user profile
        user_profile = memoryos_instance.get_user_profile_summary()
        if not user_profile or user_profile.lower() == "none":
            user_profile = "No detailed user profile available yet"
        
        # Format short-term memory
        short_term_entries = []
        for entry in retrieval_results.get("short_term_memory", []):
            short_term_entries.append(MemoryEntry(
                user_input=entry.get("user_input", ""),
                agent_response=entry.get("agent_response", ""),
                timestamp=entry.get("timestamp", ""),
                meta_info=entry.get("meta_data", {})
            ))
        
        # Format mid-term retrieved pages
        retrieved_pages = []
        for page in retrieval_results.get("retrieved_pages", [])[:max_results]:
            retrieved_pages.append(MemoryEntry(
                user_input=page.get("user_input", ""),
                agent_response=page.get("agent_response", ""),
                timestamp=page.get("timestamp", ""),
                meta_info=page.get("meta_info", {})
            ))
        
        # Format user knowledge
        user_knowledge = []
        for knowledge in retrieval_results.get("retrieved_user_knowledge", [])[:max_results]:
            user_knowledge.append(KnowledgeEntry(
                knowledge=knowledge.get("knowledge", ""),
                timestamp=knowledge.get("timestamp", ""),
                source=knowledge.get("source"),
                confidence=knowledge.get("confidence"),
                similarity_score=knowledge.get("similarity_score")
            ))
        
        # Format assistant knowledge
        assistant_knowledge = []
        for knowledge in retrieval_results.get("retrieved_assistant_knowledge", [])[:max_results]:
            assistant_knowledge.append(KnowledgeEntry(
                knowledge=knowledge.get("knowledge", ""),
                timestamp=knowledge.get("timestamp", ""),
                source=knowledge.get("source"),
                confidence=knowledge.get("confidence"),
                similarity_score=knowledge.get("similarity_score")
            ))
        
        return MemoryRetrievalResult(
            status="success",
            query=query,
            timestamp=datetime.now().isoformat(),
            user_profile=user_profile,
            short_term_memory=short_term_entries,
            short_term_count=len(short_term_entries),
            retrieved_pages=retrieved_pages,
            retrieved_user_knowledge=user_knowledge,
            retrieved_assistant_knowledge=assistant_knowledge
        )
    
    except Exception as e:
        return MemoryRetrievalResult(
            status="error",
            query=query,
            timestamp=datetime.now().isoformat(),
            user_profile=f"Error: {str(e)}",
            short_term_memory=[],
            short_term_count=0,
            retrieved_pages=[],
            retrieved_user_knowledge=[],
            retrieved_assistant_knowledge=[]
        )

@mcp.tool()
async def get_user_profile(
    user_id: str,
    include_knowledge: bool = True,
    include_assistant_knowledge: bool = False
) -> UserProfileResult:
    """
    Get comprehensive user profile and knowledge information with user isolation.
    
    Retrieves user profile analysis based on conversation history, including personality traits,
    preferences, and optionally associated knowledge entries.
    
    Args:
        user_id: The user identifier for profile retrieval (required)
        include_knowledge: Whether to include user knowledge entries in the response
        include_assistant_knowledge: Whether to include assistant knowledge entries
    
    Returns:
        UserProfileResult with user profile and optional knowledge entries
    """
    try:
        # Validate user_id
        if not user_id or not user_id.strip():
            return UserProfileResult(
                status="error",
                timestamp=datetime.now().isoformat(),
                user_id="unknown",
                assistant_id="unknown",
                user_profile="user_id is required and cannot be empty"
            )
        
        # Get user-specific MemoryOS instance
        memoryos_instance = get_memoryos_for_user(user_id.strip())
        
        # Get user profile and knowledge
        # Get user profile summary
        user_profile = memoryos_instance.get_user_profile_summary()
        if not user_profile or user_profile.lower() == "none":
            user_profile = "No detailed user profile available yet. The profile will be built as more conversations are processed."
        
        result = UserProfileResult(
            status="success",
            timestamp=datetime.now().isoformat(),
            user_id=memoryos_instance.user_id,
            assistant_id=memoryos_instance.assistant_id,
            user_profile=user_profile
        )
        
        # Include user knowledge if requested
        if include_knowledge:
            user_knowledge_entries = memoryos_instance.user_long_term_memory.get_user_knowledge()
            user_knowledge = []
            
            for entry in user_knowledge_entries:
                user_knowledge.append(KnowledgeEntry(
                    knowledge=entry.get("knowledge", ""),
                    timestamp=entry.get("timestamp", ""),
                    source=entry.get("source"),
                    confidence=entry.get("confidence")
                ))
            
            result.user_knowledge = user_knowledge
            result.user_knowledge_count = len(user_knowledge)
        
        # Include assistant knowledge if requested
        if include_assistant_knowledge:
            assistant_knowledge_entries = memoryos_instance.user_long_term_memory.get_assistant_knowledge()
            assistant_knowledge = []
            
            for entry in assistant_knowledge_entries:
                assistant_knowledge.append(KnowledgeEntry(
                    knowledge=entry.get("knowledge", ""),
                    timestamp=entry.get("timestamp", ""),
                    source=entry.get("source"),
                    confidence=entry.get("confidence")
                ))
            
            result.assistant_knowledge = assistant_knowledge
            result.assistant_knowledge_count = len(assistant_knowledge)
        
        return result
    
    except Exception as e:
        return UserProfileResult(
            status="error",
            timestamp=datetime.now().isoformat(),
            user_id=user_id.strip() if user_id else "unknown",
            assistant_id="unknown",
            user_profile=f"Error retrieving user profile: {str(e)}"
        )

if __name__ == "__main__":
    asyncio.run(main())
