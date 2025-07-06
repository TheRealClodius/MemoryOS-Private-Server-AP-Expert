# MemoryOS MCP Server

## Overview

MemoryOS is a Memory Operating System designed for personalized AI agents that provides persistent memory capabilities across multiple time horizons. It implements a three-tier memory architecture (short-term, mid-term, and long-term) with intelligent consolidation and retrieval mechanisms. The system is exposed as an MCP (Model Context Protocol) server optimized for Gemini client integration while maintaining OpenAI embedding consistency.

## System Architecture

The system follows a modular, layered architecture with clear separation of concerns:

### Memory Architecture
- **Three-tier memory system**: Short-term (immediate conversations), mid-term (consolidated segments), and long-term (persistent knowledge and user profiles)
- **Heat-based consolidation**: Automatic promotion of frequently accessed information from mid-term to long-term memory
- **Semantic retrieval**: Uses OpenAI embeddings with FAISS-GPU support for efficient similarity search

### Component Structure
- **Core Memory Classes**: Each memory tier has its own dedicated class with specific storage and retrieval logic
- **Orchestration Layer**: Main `Memoryos` class coordinates all memory operations
- **MCP Interface**: FastMCP framework provides standardized tool interface for external clients
- **Utility Layer**: Shared utilities for file operations, similarity computation, and data validation

## Key Components

### Memory Layers
1. **Short-term Memory** (`short_term.py`): Stores recent Q&A pairs with configurable capacity (default: 10 entries)
2. **Mid-term Memory** (`mid_term.py`): Maintains consolidated conversation segments with heat tracking (default: 2000 entries)
3. **Long-term Memory** (`long_term.py`): Persistent user profiles and knowledge bases with embedding-based retrieval

### Core Services
- **Memory Retriever** (`retriever.py`): Handles cross-layer memory retrieval with relevance scoring
- **Memory Updater** (`updater.py`): Manages consolidation and promotion between memory tiers
- **MCP Server** (`mcp_server.py`): Provides standardized tool interface using FastMCP framework

### Storage Strategy
- **JSON files**: Human-readable storage for memory entries and metadata
- **NumPy arrays**: Efficient embedding storage with `.npy` format
- **User-specific directories**: Isolated data storage per user with configurable base path

## Data Flow

1. **Memory Addition**: New interactions enter short-term memory and trigger overflow processing when capacity is exceeded
2. **Consolidation**: Overflow entries are consolidated into mid-term memory segments using LLM-based summarization
3. **Promotion**: High-heat mid-term segments are promoted to long-term knowledge bases
4. **Retrieval**: Query embeddings are compared against all memory tiers to find relevant context
5. **Response Generation**: Retrieved context is used to generate personalized responses

## External Dependencies

### Required Services
- **OpenAI API**: Text generation (GPT-4o-mini) and embeddings (text-embedding-3-small)
- **MCP SDK**: Anthropic's Model Context Protocol for standardized tool interfaces
- **FastMCP**: Framework for simplified MCP server implementation

### Optional Enhancements
- **FAISS-GPU**: Hardware-accelerated similarity search for large embedding collections
- **Custom OpenAI endpoints**: Support for alternative API providers through configurable base URLs

### Python Libraries
- **NumPy**: Numerical operations and embedding storage
- **Pydantic**: Data validation and structured returns
- **Standard library**: JSON, datetime, pathlib for core operations

## Deployment Strategy

### Configuration Management
- **JSON-based config**: Single `config.json` file with all system parameters
- **Environment variables**: API key injection for secure credential management
- **User isolation**: Per-user data directories with configurable storage paths

### MCP Integration
- **STDIO transport**: Standard input/output communication with MCP clients
- **Structured returns**: Pydantic models ensure consistent tool response formats
- **Error handling**: Comprehensive error catching with user-friendly messages

### Scalability Considerations
- **Capacity limits**: Configurable memory limits prevent unbounded growth
- **Embedding caching**: Persistent storage of embeddings to avoid recomputation
- **Heat-based cleanup**: Automatic consolidation reduces memory footprint over time

## Recent Changes

✅ **July 06, 2025 - SIMILARITY THRESHOLD FILTERING FIX**
- **RELEVANCE ISSUE RESOLVED**: Fixed memory retrieval returning irrelevant results
- **Problem**: System returned any matches regardless of relevance (threshold: 0.3)
- **Solution**: Implemented proper similarity filtering with 0.7 threshold across all memory layers
- **Filtering Improvements**:
  - Short-term memory: Word overlap similarity with 0.7 threshold
  - Mid-term memory: Embedding similarity threshold increased from 0.3 to 0.7
  - Long-term knowledge: User/assistant knowledge threshold increased to 0.7
  - Empty results returned when no matches meet relevance threshold
- **VERIFIED WORKING**: 
  - Relevant query "favorite color" returns matching memory (score: 1.5)
  - Irrelevant query "weather today" returns empty results
  - Partial match "blue sky" properly filtered out
- **Behavior**: Only highly relevant memories returned, improving user experience

✅ **July 06, 2025 - CRITICAL SECURITY FIX: User Data Isolation Vulnerability**
- **SECURITY VULNERABILITY RESOLVED**: Fixed major data leakage issue in API endpoints
- **Problem**: Server was using single global MemoryOS instance, causing cross-user data exposure
- **Solution**: Implemented per-user MemoryOS instances with proper isolation
- **API Security Enhancements**:
  - All endpoints now require user_id parameter for data isolation
  - Added user_id validation (cannot be empty/null)
  - Each user gets separate MemoryOS instance with isolated data storage
  - Memory retrieval now filters by user_id - NO CROSS-USER DATA LEAKAGE
- **Database Query Filtering**: Fixed retrieve_memory to filter by user_id
- **Parameter Validation**: Added comprehensive user_id validation across all endpoints
- **VERIFIED WORKING**: Tested user isolation - Alice cannot see Bob's memories
- **Security Features**: user_data_isolation, per_user_memory_instances, user_id_validation

✅ **July 06, 2025 - Dynamic User Management System Implemented**
- **USER ID ISSUES RESOLVED**: Eliminated all hardcoded user IDs and paths
- Implemented automatic UUID generation for user IDs when none provided
- Added user-specific data isolation with individual storage directories
- Created comprehensive environment variable configuration system
- Added new API endpoints: POST /api/create_user, GET /api/user_info
- Enhanced user profile endpoint to include user identification
- **VERIFIED WORKING**: Dynamic user creation and isolation operational
- Created detailed USER_MANAGEMENT.md documentation guide

✅ **July 05, 2025 - Deployment Fixes Applied Successfully**
- **DEPLOYMENT ISSUE RESOLVED**: Fixed health check failures and server startup
- Created dedicated HTTP deployment server (deploy_server.py) with FastAPI
- Added health check endpoints at / and /health for Replit deployment monitoring
- Configured proper port binding on 0.0.0.0:5000 for external access
- Implemented hybrid server mode: HTTP for deployment, MCP for local development
- Added comprehensive API endpoints: /api/add_memory, /api/retrieve_memory, /api/user_profile
- Fixed run command configuration with proper environment variable detection
- **VERIFIED WORKING**: All health checks and API endpoints operational
- Server now properly serves HTTP health checks for deployment validation

✅ **July 05, 2025 - Documentation Security Fix**
- **SECURITY FIX**: Removed API key patterns from README.md documentation
- Fixed static analysis vulnerability alert for OpenAI API key pattern in documentation
- Updated README.md to reference secure configuration template instead of showing API key examples
- Enhanced troubleshooting section to emphasize secure configuration practices
- All documentation now follows security best practices without triggering scanners

✅ **July 05, 2025 - Security Vulnerability Fix**
- **CRITICAL SECURITY FIX**: Removed hardcoded OpenAI API key from config.json
- Created secure configuration template (config.template.json)
- Added comprehensive .gitignore to prevent API key exposure
- Created SECURITY.md with detailed secure configuration guide
- Verified environment variable-based configuration works correctly
- Updated documentation to emphasize secure API key management

✅ **July 05, 2025 - Complete MCP Server Implementation & Full Testing**
- Implemented full MemoryOS MCP server with FastMCP framework
- Created three production-ready MCP tools: add_memory, retrieve_memory, get_user_profile
- Integrated complete hierarchical memory architecture (short/mid/long-term)
- Added FAISS-CPU vector search and OpenAI embeddings support
- Configured structured Pydantic responses following MCP 1.2.0+ standards
- Successfully installed all dependencies: mcp, openai, numpy, faiss-cpu, pydantic
- **COMPLETED FULL FUNCTIONALITY TESTING**: All MCP tools working with live OpenAI API
- Validated memory addition, retrieval, and user profiling with real data
- Created comprehensive deployment documentation and production guide

## Deployment Status

🚀 **DEPLOYMENT READY - ALL ISSUES RESOLVED**
- **Health Check Endpoints**: Both / and /health endpoints operational and returning proper status
- **HTTP Server**: FastAPI server running on port 5000 with external access (0.0.0.0)
- **API Endpoints**: All three endpoints working correctly:
  - POST /api/add_memory - Successfully adds memories
  - GET /api/retrieve_memory - Retrieves relevant memories with semantic search
  - GET /api/user_profile - Returns user profile information
- **Environment Configuration**: Proper port binding and environment variable handling
- **Memory Operations**: All memory tiers (short/mid/long-term) functional
- **OpenAI Integration**: Embeddings and LLM calls working with API key configuration
- **Deployment Files**: main.py, deploy_server.py, and run.py configured for production
- Ready for immediate Replit deployment

## User Preferences

```
Preferred communication style: Simple, everyday language.
```