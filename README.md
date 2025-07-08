# MemoryOS MCP Server

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green)](https://modelcontextprotocol.io/)

A production-ready Memory Operating System designed for personalized AI agents, implementing a three-tier hierarchical memory architecture inspired by computer science memory management principles. Provides persistent memory capabilities through MCP (Model Context Protocol) server integration.

## 🧠 Overview

MemoryOS is a sophisticated memory management system for AI agents that mimics how human memory works across different time horizons. The system automatically stores, organizes, and retrieves conversation history while building comprehensive user profiles and knowledge bases.

**Key Differentiators:**
- **Redis-style Performance**: Sub-millisecond access to recent conversations
- **Intelligent Memory Promotion**: Heat-based consolidation from short-term to long-term memory
- **Semantic Understanding**: Uses OpenAI embeddings for context-aware retrieval
- **Complete User Isolation**: Each user has separate, secure data storage
- **Production-Ready**: Deployed with proper health checks and error handling

## ✨ Features

### Memory Architecture
- **🚀 Redis-style Short-term Memory**: Sub-millisecond FIFO storage using Python deque for recent conversations
- **📊 Indexed Mid-term Memory**: Heat-based consolidation with embedding search for conversation segments  
- **🧠 Persistent Long-term Memory**: User profiles and knowledge bases with semantic search
- **🔄 Intelligent Memory Promotion**: Automatic consolidation based on access patterns and recency

### Advanced Capabilities
- **🔍 Semantic Memory Retrieval**: Vector similarity search using OpenAI embeddings and FAISS
- **👤 Automatic User Profiling**: Builds comprehensive personality profiles from conversation history
- **📚 Knowledge Extraction**: Identifies and stores important facts about users and topics
- **🎯 Relevance Filtering**: Smart threshold-based filtering to return only highly relevant memories
- **⚡ Heat-based Optimization**: Frequently accessed information promotes to faster storage tiers

### Production Features
- **🔐 COMPLETE USER ISOLATION**: Each API key gets isolated memory instances - Alice's conversations never visible to Bob
- **🗂️ File System Isolation**: Users get separate data directories (`./memoryos_data/user_id/`)
- **🔑 Session-Based Security**: Each session maps to specific user ID with no cross-contamination
- **🌐 HTTP API Deployment**: Ready for production deployment with health checks
- **🔧 Pure MCP 2.0 Protocol**: Standards-compliant JSON-RPC implementation for seamless client integration
- **📈 Comprehensive Monitoring**: Health checks, performance metrics, and detailed logging
- **🛡️ Security Hardened**: Environment variable configuration, input validation, and data protection

## 🏗️ System Architecture

MemoryOS implements a sophisticated three-tier memory hierarchy inspired by operating system memory management:

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryOS Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  🚀 SHORT-TERM MEMORY (Redis-style)                       │
│  ├─ Python deque with maxlen (FIFO)                       │
│  ├─ Sub-millisecond access time                           │
│  ├─ Recent conversations (default: 10 entries)            │
│  └─ Automatic overflow to mid-term                        │
├─────────────────────────────────────────────────────────────┤
│  📊 MID-TERM MEMORY (Heat-based)                          │
│  ├─ JSON segments + numpy embeddings                      │
│  ├─ Heat tracking for access patterns                     │
│  ├─ Consolidation of conversation threads                 │
│  └─ Promotion to long-term when hot                       │
├─────────────────────────────────────────────────────────────┤
│  🧠 LONG-TERM MEMORY (Persistent)                         │
│  ├─ User profile & personality analysis                   │
│  ├─ Knowledge bases (user + assistant)                    │
│  ├─ Semantic embeddings for search                        │
│  └─ FAISS vector search optimization                      │
└─────────────────────────────────────────────────────────────┘
```

### Memory Flow & Data Processing

1. **Input Stage**: New conversations enter short-term memory (Redis-style deque)
2. **Overflow Processing**: When capacity is reached, oldest entries move to mid-term storage
3. **Heat Analysis**: Mid-term segments track access frequency, recency, and interaction depth
4. **Smart Promotion**: High-heat segments get analyzed by LLM for long-term insights
5. **Knowledge Extraction**: Important facts become searchable knowledge entries
6. **Profile Building**: User characteristics and preferences are continuously updated

### Performance Characteristics

| Memory Tier | Access Time | Storage Type | Capacity | Search Method |
|-------------|-------------|--------------|----------|---------------|
| Short-term  | < 1ms       | In-memory deque | 10 entries | Linear scan |
| Mid-term    | < 50ms      | JSON + embeddings | 2000 segments | Embedding similarity |
| Long-term   | < 100ms     | JSON + FAISS | 100 knowledge items | Vector search |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key with available quota
- Claude Desktop (for MCP client integration) or HTTP client for API access

### Installation & Setup

1. **Clone the repository or download MemoryOS files**

2. **Install dependencies:**
   ```bash
   pip install mcp openai numpy faiss-cpu pydantic fastapi uvicorn
   ```

3. **Configure your OpenAI API key** (choose one method):

   **Option A: Environment Variable (Recommended)**
   ```bash
   export OPENAI_API_KEY="your_actual_openai_api_key"
   ```

   **Option B: Configuration File**
   ```bash
   cp config.template.json config.json
   # Edit config.json to add your OpenAI API key
   ```
   
   **⚠️ IMPORTANT**: Never commit API keys to version control. See security section below.

4. **Test the installation:**
   ```bash
   # Test Redis-style memory implementation
   python test_redis_style_memory.py
   
   # Test complete memory tier integration  
   python test_complete_memory_tiers.py
   
   # Test with live OpenAI API (requires API key)
   python test_full_functionality.py
   ```

5. **Start the server:**
   ```bash
   # Start pure MCP 2.0 server
   python mcp_server.py
   
   # Or use main entry point
   python main.py
   ```

### Deployment Options

**Pure MCP 2.0 Remote Server**
- Runs on port 5000 with JSON-RPC 2.0 over HTTP
- Standards-compliant MCP 2.0 protocol implementation
- Bearer token authentication for secure access
- Complete user isolation with per-user memory instances
- Health checks and monitoring endpoints
- Ready for production deployment

## 🔧 MCP 2.0 Protocol Reference

### Health Check Endpoints

```bash
# Basic server information
curl http://localhost:5000/

# Detailed health status
curl http://localhost:5000/health
```

### MCP 2.0 JSON-RPC API

All MCP operations use the `/mcp/` endpoint with Bearer authentication:

#### Authentication
```bash
# All requests require Bearer token
-H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
```

#### Initialize MCP Session
```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {"tools": {}},
      "clientInfo": {"name": "CustomClient", "version": "1.0.0"}
    },
    "id": 1
  }'
```

#### List Available Tools
```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
  }'
```

#### Add Memory (MCP 2.0 Format)
```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "add_memory",
      "arguments": {
        "params": {
          "user_input": "I love working with Python",
          "agent_response": "That'\''s great! Python is excellent for data science and AI projects.",
          "user_id": "alice_2024"
        }
      }
    },
    "id": 3
  }'
```

#### Retrieve Memory (MCP 2.0 Format)
```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "retrieve_memory",
      "arguments": {
        "params": {
          "query": "Python programming",
          "user_id": "alice_2024"
        }
      }
    },
    "id": 4
  }'
```

#### Get User Profile (MCP 2.0 Format)
```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_user_profile",
      "arguments": {
        "params": {
          "user_id": "alice_2024"
        }
      }
    },
    "id": 5
  }'
```

### Response Examples

**MCP 2.0 Response Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\":\"success\",\"message\":\"Memory added successfully\",\"user_id\":\"alice_2024\",\"timestamp\":\"2025-07-08T10:30:00Z\"}"
      }
    ],
    "isError": false
  }
}
```

**Memory Retrieval Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\":\"success\",\"query\":\"Python programming\",\"user_id\":\"alice_2024\",\"result\":{\"user_profile\":\"Alice is interested in programming...\",\"short_term_memory\":[...]}}"
      }
    ],
    "isError": false
  }
}
```

## 💻 MCP Integration Guide

### Claude Desktop Setup

1. **Locate your Claude Desktop configuration file:**
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Add MemoryOS server configuration:**

```json
{
  "mcpServers": {
    "memoryos": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    }
  }
}
```

**Important Notes:**
- Use the full absolute path to `mcp_server.py`
- Replace `your_openai_api_key_here` with your actual OpenAI API key
- Ensure Python is available in your system PATH

3. **Restart Claude Desktop** to load the new configuration

### Verification

1. **Look for the hammer icon (🔨)** in Claude Desktop - this indicates MCP tools are available
2. **Start a conversation** - MemoryOS will automatically handle memory operations
3. **Test memory retention** by referencing previous conversations

### Available MCP Tools

MemoryOS provides three MCP tools that Claude Desktop can automatically use:

#### `add_memory`
Stores conversation pairs in the memory system.
- **Automatically called** when conversations occur
- **Parameters**: user_input, agent_response, timestamp, metadata
- **Returns**: Success confirmation with storage details

#### `retrieve_memory`  
Searches memory for relevant information.
- **Automatically called** when context is needed
- **Parameters**: query, relationship_with_user, style_hint, max_results
- **Returns**: Relevant conversation history and knowledge

#### `get_user_profile`
Retrieves user profile and knowledge summary.
- **Automatically called** for personalization
- **Parameters**: include_knowledge, include_assistant_knowledge
- **Returns**: User personality analysis and knowledge entries

## 📊 Usage Examples

### Example 1: Learning and Remembering Preferences

```
User: "I prefer concise explanations and I'm working on a machine learning project"
Claude: [Uses add_memory to store this preference and project information]

[Later in conversation]
User: "Can you explain neural networks?"
Claude: [Uses retrieve_memory to recall preference for concise explanations]
       "Here's a concise explanation of neural networks for your ML project..."
```

### Example 2: Building Context Over Time

```
Day 1:
User: "I'm learning Python for data analysis"
Claude: [Stores: user is learning Python, interested in data analysis]

Day 3:
User: "What's a good visualization library?"
Claude: [Retrieves: user works with Python + data analysis]
       "For Python data analysis, I recommend matplotlib or seaborn..."

Day 7:
User: "Remember what I'm working on?"
Claude: [Retrieves comprehensive history]
       "You're learning Python for data analysis. We've discussed 
        visualization libraries like matplotlib..."
```

### Example 3: Project Continuity

```
User: "I'm building a web scraper for financial data"
Claude: [add_memory stores project details]

[Next session]
User: "How do I handle rate limiting?"
Claude: [retrieve_memory finds web scraper context]
       "For your financial data scraper, here are rate limiting strategies..."
```

## 🔒 Security & Privacy

### Data Privacy
- **Local Storage Only**: All conversation data stays on your machine
- **No Data Transmission**: Only search queries sent to OpenAI, never stored conversations
- **User Isolation**: Each user has completely separate data directories
- **Secure Configuration**: Environment variable support for API keys

### Security Features
- **Input Validation**: All API endpoints validate user_id and parameters
- **Path Sanitization**: Prevents directory traversal attacks
- **Error Handling**: Detailed logging without exposing sensitive information
- **Rate Limiting**: Built-in protection against excessive API usage

### Best Practices
```bash
# Use environment variables for API keys
export OPENAI_API_KEY="your_key_here"

# Set secure permissions on data directory
chmod 700 memoryos_data/

# Regular backup of important conversations
cp -r memoryos_data/ backup_$(date +%Y%m%d)/
```

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Required
export OPENAI_API_KEY="your_openai_api_key"

# Optional
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Custom endpoint
export MEMORYOS_DATA_PATH="./secure_storage"        # Custom data path
export MEMORYOS_LOG_LEVEL="INFO"                    # Logging level
```

### Configuration File Options
```json
{
  "user_id": "your_unique_id",
  "openai_base_url": "https://api.openai.com/v1",
  "data_storage_path": "./memoryos_data",
  "assistant_id": "mcp_assistant",
  "llm_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "short_term_capacity": 10,
  "mid_term_capacity": 2000,
  "long_term_knowledge_capacity": 100,
  "retrieval_queue_capacity": 7,
  "mid_term_heat_threshold": 5.0
}
```

### Performance Tuning

#### Memory Capacity Optimization
```json
{
  "short_term_capacity": 5,     // Faster, less context
  "mid_term_capacity": 1000,    // Reduce for faster search
  "long_term_knowledge_capacity": 50  // Smaller knowledge base
}
```

#### Search Performance
```json
{
  "retrieval_queue_capacity": 3,  // Fewer results, faster response
  "mid_term_heat_threshold": 3.0  // More aggressive promotion
}
```

### Multi-User Deployment
```bash
# User-specific configurations
export MEMORYOS_USER_ID="alice"
export MEMORYOS_DATA_PATH="./users/alice"

# Run server (users isolated automatically by user_id in MCP calls)
python mcp_server.py
```

## 🐛 Troubleshooting

### Common Issues

#### Server Won't Start
```bash
# Check Python version
python --version  # Should be 3.11+

# Verify dependencies
pip list | grep -E "(mcp|openai|numpy|faiss|pydantic)"

# Test configuration
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT_SET'))"
```

#### API Key Issues
```bash
# Verify API key format
echo $OPENAI_API_KEY | grep -E "^sk-[a-zA-Z0-9]+"

# Test OpenAI connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models" | head -10
```

#### Memory Issues
```bash
# Check data directory permissions
ls -la memoryos_data/

# Verify disk space
df -h

# Check memory usage
python test_redis_style_memory.py
```

#### Claude Desktop Integration
```bash
# Validate JSON configuration
python -m json.tool claude_desktop_config.json

# Check file paths
ls -la /absolute/path/to/mcp_server.py

# Test MCP server directly  
python mcp_server.py
```

### Performance Diagnostics
```python
# Test memory tier performance
python test_complete_memory_tiers.py

# Monitor API usage
tail -f memoryos_data/logs/api_usage.log

# Check embedding generation speed
python test_embedding_issue.py
```

### Debug Mode
```bash
# Enable detailed logging
export MEMORYOS_LOG_LEVEL="DEBUG" 
python mcp_server.py

# Save debug output
python mcp_server.py 2>&1 | tee debug.log
```

## 📁 Data Storage Structure

MemoryOS creates a hierarchical directory structure for organized data storage:

```
memoryos_data/
├── user_alice/
│   ├── short_term/
│   │   ├── memory.json          # Redis-style conversation deque
│   │   └── overflow.json        # Evicted entries for consolidation
│   ├── mid_term/
│   │   ├── memory.json          # Conversation segments with heat
│   │   └── embeddings.npy       # Segment embeddings for search
│   └── long_term/
│       ├── user_profile.json    # Personality and preferences
│       ├── user_knowledge.json  # Facts about the user
│       ├── assistant_knowledge_mcp_assistant.json  # Assistant knowledge
│       ├── user_knowledge_embeddings.npy           # User knowledge vectors
│       └── assistant_knowledge_embeddings_mcp_assistant.npy
└── user_bob/
    └── [same structure for different user]
```

### Data Format Examples

**Short-term Memory Entry:**
```json
{
  "user_input": "What's the weather like?",
  "agent_response": "I don't have access to real-time weather data...",
  "timestamp": "2025-07-06T19:30:45.123456",
  "meta_data": {},
  "access_count": 0,
  "last_accessed": "2025-07-06T19:30:45.123456"
}
```

**Mid-term Memory Segment:**
```json
{
  "id": "segment_abc123",
  "user_id": "alice",
  "timestamp": "2025-07-06T19:30:45.123456",
  "qa_pairs": [...],
  "summary": "Discussion about Python programming and data analysis",
  "themes": ["programming", "python", "data-analysis"],
  "heat": 3.5,
  "access_count": 7,
  "last_accessed": "2025-07-06T20:15:30.789012"
}
```

## 🚀 Deployment Guide

### Development
```bash
# Local development
python mcp_server.py

# Test with sample data
python simple_memory_test.py
```

### Production
```bash
# Production HTTP server
python mcp_server.py

# With process management
python mcp_server.py

# Docker deployment (if needed)
docker build -t memoryos .
docker run -p 5000:5000 -e OPENAI_API_KEY=$OPENAI_API_KEY memoryos
```

### Health Monitoring
```bash
# Check server health
curl http://localhost:5000/health

# Monitor performance
# Test MCP 2.0 authentication
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}'

# Automated health checks
while true; do
  curl -s http://localhost:5000/ | grep -q "healthy" && echo "OK" || echo "ERROR"
  sleep 30
done
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test files for examples
3. Ensure your OpenAI API key has sufficient quota
4. Verify all dependencies are correctly installed

The MemoryOS system is designed to be robust and self-healing, with comprehensive error handling and logging to help diagnose any issues.

