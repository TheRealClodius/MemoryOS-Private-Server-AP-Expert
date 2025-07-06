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

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"ERROR: Failed to import MCP SDK. Please install: pip install mcp", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

# Import FastAPI for HTTP health check endpoint
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
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

@mcp.tool()
async def add_memory(
    user_input: str,
    agent_response: str,
    timestamp: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None
) -> MemoryOperationResult:
    """
    Add a new memory entry to MemoryOS system.
    
    Stores conversation pairs (user input + agent response) in the hierarchical memory system
    for building persistent dialogue history and contextual understanding.
    
    Args:
        user_input: The user's input, question, or statement
        agent_response: The assistant's response to the user input
        timestamp: Optional timestamp in ISO format (uses current time if not provided)
        meta_data: Optional metadata dictionary for additional context
    
    Returns:
        MemoryOperationResult with operation status and details
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return MemoryOperationResult(
            status="error",
            message="MemoryOS is not initialized. Check configuration and restart server.",
            timestamp=datetime.now().isoformat()
        )
    
    try:
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
        
        # Add memory to MemoryOS
        result = memoryos_instance.add_memory(
            user_input=user_input.strip(),
            agent_response=agent_response.strip(),
            timestamp=timestamp,
            meta_data=meta_data
        )
        
        if result.get("status") == "success":
            return MemoryOperationResult(
                status="success",
                message="Memory successfully added to MemoryOS hierarchical storage",
                timestamp=datetime.now().isoformat(),
                details={
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
    relationship_with_user: str = "assistant",
    style_hint: str = "",
    max_results: int = 10
) -> MemoryRetrievalResult:
    """
    Retrieve relevant memories and context from MemoryOS system.
    
    Searches across all memory layers (short-term, mid-term, long-term) using semantic similarity
    to find relevant conversation history, user knowledge, and assistant knowledge.
    
    Args:
        query: The search query or topic to find relevant memories for
        relationship_with_user: Relationship context (assistant, friend, colleague, etc.)
        style_hint: Style preference for response formatting
        max_results: Maximum number of results to return per category (default: 10)
    
    Returns:
        MemoryRetrievalResult with comprehensive memory context
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return MemoryRetrievalResult(
            status="error",
            query=query,
            timestamp=datetime.now().isoformat(),
            user_profile="MemoryOS not initialized",
            short_term_memory=[],
            short_term_count=0,
            retrieved_pages=[],
            retrieved_user_knowledge=[],
            retrieved_assistant_knowledge=[]
        )
    
    try:
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
    include_knowledge: bool = True,
    include_assistant_knowledge: bool = False
) -> UserProfileResult:
    """
    Get comprehensive user profile and knowledge information.
    
    Retrieves user profile analysis based on conversation history, including personality traits,
    preferences, and optionally associated knowledge entries.
    
    Args:
        include_knowledge: Whether to include user knowledge entries in the response
        include_assistant_knowledge: Whether to include assistant knowledge entries
    
    Returns:
        UserProfileResult with user profile and optional knowledge entries
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return UserProfileResult(
            status="error",
            timestamp=datetime.now().isoformat(),
            user_id="unknown",
            assistant_id="unknown",
            user_profile="MemoryOS is not initialized. Check configuration and restart server."
        )
    
    try:
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
            user_id=memoryos_instance.user_id if memoryos_instance else "unknown",
            assistant_id=memoryos_instance.assistant_id if memoryos_instance else "unknown",
            user_profile=f"Error retrieving user profile: {str(e)}"
        )

async def init_server():
    """Initialize the MemoryOS server"""
    global memoryos_instance
    
    # Load configuration
    print("Loading MemoryOS configuration...", file=sys.stderr)
    config = load_config()
    
    # Initialize MemoryOS
    print(f"Initializing MemoryOS for user: {config['user_id']}", file=sys.stderr)
    memoryos_instance = init_memoryos(config)
    
    # Verify initialization by checking memory stats
    try:
        stats = memoryos_instance.get_memory_stats()
        print(f"MemoryOS initialization verified:", file=sys.stderr)
        print(f"  Short-term memories: {stats.get('short_term', {}).get('total_entries', 0)}", file=sys.stderr)
        print(f"  User data path: {memoryos_instance.data_storage_path}/{memoryos_instance.user_id}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not verify MemoryOS initialization: {e}", file=sys.stderr)
    
    print(f"MemoryOS MCP Server started successfully", file=sys.stderr)
    print(f"User: {config['user_id']}, Assistant: {config['assistant_id']}", file=sys.stderr)
    print(f"Data storage: {config['data_storage_path']}", file=sys.stderr)
    print(f"LLM model: {config['llm_model']}, Embedding model: {config['embedding_model']}", file=sys.stderr)
    
    return config

async def run_mcp_server():
    """Run the MCP server with stdio transport"""
    try:
        await init_server()
        print("Starting MCP server with stdio transport...", file=sys.stderr)
        
        # Run MCP server with stdio transport
        await mcp.run(transport="stdio")
    
    except KeyboardInterrupt:
        print("\nMCP server interrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"MCP server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_health_check_app():
    """Create FastAPI app for health checks"""
    if not FASTAPI_AVAILABLE:
        return None
    
    app = FastAPI(title="MemoryOS MCP Server", version="1.0.0")
    
    @app.get("/")
    async def health_check():
        """Health check endpoint for deployment"""
        try:
            # Initialize server if not already done
            if memoryos_instance is None:
                await init_server()
            
            return JSONResponse({
                "status": "healthy",
                "service": "MemoryOS MCP Server",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
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
        """Detailed health check"""
        try:
            if memoryos_instance is None:
                await init_server()
            
            # Test basic functionality
            stats = memoryos_instance.get_memory_stats()
            
            return JSONResponse({
                "status": "healthy",
                "service": "MemoryOS MCP Server",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "memory_stats": stats,
                "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
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
    
    return app

async def run_http_server():
    """Run HTTP server for health checks"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available, cannot run HTTP server", file=sys.stderr)
        return
    
    app = create_health_check_app()
    if app is None:
        return
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "5000"))
    
    print(f"Starting HTTP health check server on port {port}...", file=sys.stderr)
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Main server function"""
    try:
        # Check if we should run HTTP server (for deployment) or MCP server (for local use)
        mode = os.getenv("SERVER_MODE", "mcp").lower()
        
        if mode == "http" or os.getenv("PORT"):
            # Run HTTP server for deployment
            await run_http_server()
        else:
            # Run MCP server for local use
            await run_mcp_server()
            
    except KeyboardInterrupt:
        print("\nServer interrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
