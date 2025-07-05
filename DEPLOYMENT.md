# MemoryOS MCP Server Deployment Guide

## Overview

MemoryOS is a complete Memory Operating System implemented as an MCP (Model Context Protocol) server. It provides persistent memory capabilities across multiple time horizons using a three-tier architecture (short-term, mid-term, and long-term memory) with intelligent consolidation and semantic retrieval.

## Features

✅ **Complete MCP Server Implementation**
- Three MCP tools: `add_memory`, `retrieve_memory`, `get_user_profile`
- Structured Pydantic responses following MCP 1.2.0+ standards
- FastMCP framework integration for optimal performance

✅ **Hierarchical Memory Architecture**
- Short-term memory: Recent conversation pairs (configurable capacity)
- Mid-term memory: Consolidated conversation segments with heat tracking
- Long-term memory: Persistent user profiles and knowledge bases

✅ **Advanced AI Integration**
- OpenAI embeddings (text-embedding-3-small) for semantic similarity
- FAISS-CPU vector search for efficient retrieval
- GPT-4o-mini for text processing and consolidation
- Optimized for Gemini client integration

✅ **Production-Ready Features**
- User-specific data isolation
- Configurable memory limits and thresholds
- Comprehensive error handling
- JSON-based configuration with environment variable fallbacks

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies (already installed in this environment)
pip install mcp openai numpy faiss-cpu pydantic
```

### 2. Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

Or update `config.json`:

```json
{
  "openai_api_key": "your_openai_api_key_here"
}
```

### 3. Start the MCP Server

```bash
python mcp_server.py
```

The server will:
- Load configuration from `config.json` and environment variables
- Initialize the MemoryOS system
- Start listening for MCP client connections via stdio

## MCP Client Integration

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "memoryos": {
      "command": "python",
      "args": ["/path/to/your/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    }
  }
}
```

### Available Tools

#### 1. `add_memory`
Stores conversation pairs in the hierarchical memory system.

**Parameters:**
- `user_input` (string): User's input or question
- `agent_response` (string): Assistant's response
- `timestamp` (optional string): ISO timestamp
- `meta_data` (optional object): Additional metadata

**Returns:** MemoryOperationResult with status and details

#### 2. `retrieve_memory` 
Retrieves relevant memories using semantic similarity search.

**Parameters:**
- `query` (string): Search query or topic
- `relationship_with_user` (string, default: "assistant"): Relationship context
- `style_hint` (string, default: ""): Style preference
- `max_results` (int, default: 10): Maximum results per category

**Returns:** MemoryRetrievalResult with comprehensive memory context

#### 3. `get_user_profile`
Gets user profile and knowledge information.

**Parameters:**
- `include_knowledge` (bool, default: true): Include user knowledge entries
- `include_assistant_knowledge` (bool, default: false): Include assistant knowledge

**Returns:** UserProfileResult with profile and knowledge data

## Configuration Options

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `user_id` | "default_user" | Unique user identifier |
| `assistant_id` | "mcp_assistant" | Assistant identifier |
| `data_storage_path` | "./memoryos_data" | Data storage directory |
| `llm_model` | "gpt-4o-mini" | OpenAI model for text processing |
| `embedding_model` | "text-embedding-3-small" | OpenAI embedding model |

### Memory Capacity Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `short_term_capacity` | 10 | Max conversation pairs in short-term |
| `mid_term_capacity` | 2000 | Max segments in mid-term memory |
| `long_term_knowledge_capacity` | 100 | Max knowledge entries per category |
| `retrieval_queue_capacity` | 7 | Max results per retrieval category |
| `mid_term_heat_threshold` | 5.0 | Heat threshold for long-term promotion |

### Environment Variables

All configuration options can be overridden with environment variables:

```bash
export MEMORYOS_USER_ID="your_user_id"
export OPENAI_API_KEY="your_api_key"
export MEMORYOS_DATA_PATH="/custom/data/path"
export MEMORYOS_LLM_MODEL="gpt-4"
export MEMORYOS_EMBEDDING_MODEL="text-embedding-3-large"
# ... and more
```

## Data Storage

MemoryOS creates user-specific directories under the configured data path:

```
memoryos_data/
└── user_id/
    ├── short_term_memory.json
    ├── mid_term_memory.json
    ├── mid_term_embeddings.npy
    ├── user_profile.json
    ├── user_knowledge.json
    ├── user_knowledge_embeddings.npy
    ├── assistant_knowledge.json
    └── assistant_knowledge_embeddings.npy
```

## Testing

Run the component test suite:

```bash
python simple_test.py
```

This validates:
- MemoryOS import and initialization
- MCP server structure
- Pydantic model validation
- Tool function signatures

## Architecture

### Memory Flow

1. **Addition**: New interactions enter short-term memory
2. **Consolidation**: Overflow entries are consolidated into mid-term segments
3. **Promotion**: High-heat segments are promoted to long-term knowledge
4. **Retrieval**: Semantic search across all memory layers
5. **Response**: Context-aware response generation

### Integration Points

- **OpenAI API**: Text generation and embeddings
- **FAISS**: Vector similarity search
- **MCP Protocol**: Standardized tool interface
- **FastMCP**: Simplified server implementation

## Troubleshooting

### Common Issues

1. **"OpenAI API key is required"**
   - Set `OPENAI_API_KEY` environment variable
   - Or add to `config.json`

2. **Import errors**
   - Ensure all dependencies are installed: `pip install mcp openai numpy faiss-cpu pydantic`

3. **Permission errors**
   - Check write permissions for data storage directory
   - Ensure user has access to create subdirectories

4. **Memory errors**
   - Check available disk space for data storage
   - Consider reducing memory capacity settings

### Debug Mode

Add debug logging by setting environment variable:

```bash
export MEMORYOS_DEBUG=true
python mcp_server.py
```

## Production Deployment

For production deployment:

1. **Security**: Store API keys securely (environment variables, not config files)
2. **Monitoring**: Monitor memory usage and API call limits
3. **Backup**: Regular backup of user data directories
4. **Scaling**: Consider user-specific server instances for high loads
5. **Updates**: Test new versions with backup/restore procedures

## Support

For issues or questions:
- Check the configuration in `config.json`
- Review the test suite output from `simple_test.py`
- Verify OpenAI API key and quota
- Ensure proper file permissions for data storage

The MemoryOS MCP server is ready for production use with proper configuration and API credentials.