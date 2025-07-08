# MemoryOS Pure MCP 2.0 Remote Server Deployment Guide

## Overview

MemoryOS implements a **pure MCP 2.0 remote server** using **direct JSON-RPC 2.0 specification**, allowing multiple applications to connect to a single MemoryOS instance over HTTP with **comprehensive security features** and **standards-compliant protocol implementation**.

## Architecture

```
┌─────────────────────┐     MCP 2.0 JSON-RPC ┌──────────────────────┐
│   MCP Client 1      │ ←------------------→ │                      │
│   (Claude Desktop)  │                      │                      │
└─────────────────────┘                      │                      │
                                             │   MemoryOS Pure      │
┌─────────────────────┐     MCP 2.0 JSON-RPC │   MCP 2.0 Server     │
│   MCP Client 2      │ ←------------------→ │   (Standards-        │
│   (Custom App)      │                      │    Compliant)        │
└─────────────────────┘                      │                      │
                                             │   🔐 Bearer Auth     │
┌─────────────────────┐     MCP 2.0 JSON-RPC │   🛡️  Security       │
│   MCP Client N      │ ←------------------→ │   👥 User Isolation  │
│   (Mobile App)      │                      │   📊 Memory Mgmt     │
└─────────────────────┘                      └──────────────────────┘
```

## 🔐 Security Features

### **1. Bearer Token Authentication**

The server uses **Bearer token authentication** following security best practices:

```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

### **2. API Key Management**

#### **Environment Variable Configuration**
```bash
# Set custom API key
export MCP_API_KEY="your-secure-api-key"

# Server will use this key for authentication
python mcp_server.py
```

#### **Default Key**
If no environment variable is set, the server uses a default secure key:
- **Default API Key**: `77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4`

### **3. User Isolation**

**Complete user isolation** prevents data leakage between different clients:

- **Per-User Memory Instances**: Each API session gets isolated MemoryOS instance
- **File System Isolation**: Users get separate data directories (`./memoryos_data/user_id/`)
- **Memory Layer Isolation**: Complete separation of short-term, mid-term, and long-term memory
- **No Cross-User Data Leakage**: Each user_id maintains completely separate memory context

## 🚀 Deployment

### **Quick Start**

```bash
# 1. Install dependencies (if not already installed)
pip install fastapi uvicorn pydantic

# 2. Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"

# 3. Optional: Set custom MCP API key
export MCP_API_KEY="your-secure-mcp-api-key"

# 4. Run the server
python mcp_server.py
```

The server will start on `http://0.0.0.0:5000`

### **Production Deployment**

```bash
# Set production environment variables
export OPENAI_API_KEY="your-openai-api-key"
export MCP_API_KEY="production-secure-key-here"

# Run with production settings
python mcp_server.py
```

## 🔧 Client Configuration

### **Claude Desktop Configuration**

Create or update `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memoryos-remote": {
      "command": "node",
      "args": ["/path/to/mcp-client.js"],
      "env": {
        "MCP_SERVER_URL": "http://localhost:5000/mcp/",
        "MCP_API_KEY": "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
      }
    }
  }
}
```

### **Custom Python Client**

```python
import httpx
import json

class MemoryOSPureMCPClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def initialize(self):
        """Initialize MCP session"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "clientInfo": {"name": "CustomClient", "version": "1.0.0"}
                    },
                    "id": 1
                }
            )
            return response.json()
    
    async def list_tools(self):
        """List available tools"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": 2
                }
            )
            return response.json()
    
    async def add_memory(self, user_input: str, agent_response: str, user_id: str):
        """Add memory using MCP 2.0 nested parameter format"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "add_memory",
                        "arguments": {
                            "params": {  # MCP 2.0 nested format
                                "user_input": user_input,
                                "agent_response": agent_response,
                                "user_id": user_id
                            }
                        }
                    },
                    "id": 3
                }
            )
            return response.json()
    
    async def retrieve_memory(self, query: str, user_id: str):
        """Retrieve memory using MCP 2.0 nested parameter format"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "retrieve_memory",
                        "arguments": {
                            "params": {  # MCP 2.0 nested format
                                "query": query,
                                "user_id": user_id
                            }
                        }
                    },
                    "id": 4
                }
            )
            return response.json()

# Usage Example
async def main():
    client = MemoryOSPureMCPClient(
        "http://localhost:5000", 
        "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4"
    )
    
    # Initialize session
    init_result = await client.initialize()
    print("Initialized:", init_result)
    
    # Add memory
    add_result = await client.add_memory(
        "What's the weather today?",
        "It's sunny and 75°F",
        "user_123"
    )
    print("Added memory:", add_result)
    
    # Retrieve memory
    retrieve_result = await client.retrieve_memory("weather", "user_123")
    print("Retrieved memory:", retrieve_result)
```

## 📚 Available Tools

### **add_memory**
Add new memory entries with user isolation.

**MCP 2.0 Format:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": {
      "params": {
        "user_input": "What is the weather like?",
        "agent_response": "The weather is sunny and 75°F",
        "user_id": "user_123",
        "timestamp": "2025-07-08T10:30:00Z",
        "meta_data": {"location": "San Francisco"}
      }
    }
  },
  "id": 1
}
```

**Required Parameters:**
- `user_input`: The user's input or question
- `agent_response`: The agent's response
- `user_id`: User identifier for memory isolation

**Optional Parameters:**
- `timestamp`: ISO format timestamp
- `meta_data`: Additional metadata dictionary

### **retrieve_memory**
Retrieve relevant memories with user isolation.

**MCP 2.0 Format:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "retrieve_memory",
    "arguments": {
      "params": {
        "query": "weather information",
        "user_id": "user_123",
        "relationship_with_user": "assistant",
        "style_hint": "casual",
        "max_results": 10
      }
    }
  },
  "id": 2
}
```

**Required Parameters:**
- `query`: Search query string
- `user_id`: User identifier for memory isolation

**Optional Parameters:**
- `relationship_with_user`: Relationship context (default: "assistant")
- `style_hint`: Style preference (default: "")
- `max_results`: Maximum results to return (default: 10)

### **get_user_profile**
Get comprehensive user profile and knowledge.

**MCP 2.0 Format:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_user_profile",
    "arguments": {
      "params": {
        "user_id": "user_123",
        "include_knowledge": true,
        "include_assistant_knowledge": false
      }
    }
  },
  "id": 3
}
```

**Required Parameters:**
- `user_id`: User identifier

**Optional Parameters:**
- `include_knowledge`: Include user knowledge entries (default: true)
- `include_assistant_knowledge`: Include assistant knowledge (default: false)

## 🛡️ Protocol Compliance

### **MCP 2.0 JSON-RPC Implementation**

The server implements pure MCP 2.0 specification:

1. **Initialize Handshake:**
   ```json
   {"jsonrpc": "2.0", "method": "initialize", "params": {...}, "id": 1}
   ```

2. **Initialized Notification:**
   ```json
   {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
   ```

3. **Tools Listing:**
   ```json
   {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
   ```

4. **Tool Calls:**
   ```json
   {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "...", "arguments": {...}}, "id": 3}
   ```

### **Parameter Format Support**

The server supports both parameter formats:

1. **MCP 2.0 Nested Format (Recommended):**
   ```json
   "arguments": {"params": {"user_input": "...", "user_id": "..."}}
   ```

2. **Direct Format (Backward Compatibility):**
   ```json
   "arguments": {"user_input": "...", "user_id": "..."}
   ```

## 📊 API Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/` | No | Server information |
| `GET` | `/health` | No | Health check |
| `POST` | `/mcp/` | Yes | MCP 2.0 JSON-RPC |

## 🔐 Security Best Practices

### **Production Recommendations**

1. **Use Strong API Keys**
   ```bash
   # Generate secure API key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Environment Variables**
   ```bash
   # Never hardcode keys
   export MCP_API_KEY="production-secure-key"
   export OPENAI_API_KEY="your-openai-key"
   ```

3. **HTTPS in Production**
   Use a reverse proxy with SSL termination:
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       location /mcp/ {
           proxy_pass http://localhost:5000/mcp/;
           proxy_set_header Authorization $http_authorization;
       }
   }
   ```

4. **Firewall Configuration**
   ```bash
   # Only allow necessary ports
   sudo ufw allow 443/tcp
   sudo ufw deny 5000/tcp  # Block direct access
   ```

## 🧪 Testing

### **Health Check**
```bash
curl http://localhost:5000/health
```

### **Authentication Test**
```bash
curl -X POST http://localhost:5000/mcp/ \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

### **Tool Test**
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
          "user_input": "Test message",
          "agent_response": "Test response",
          "user_id": "test_user"
        }
      }
    },
    "id": 1
  }'
```

## 🔄 Migration from FastMCP

If migrating from the old FastMCP implementation:

1. **Update Server:**
   - Use `mcp_server.py` instead of `deploy_server.py`
   - Port changes from 3000 to 5000
   - Authentication method changes to Bearer token

2. **Update Client Code:**
   - Add `user_id` parameter to all tool calls
   - Use MCP 2.0 nested parameter format
   - Update authentication headers

3. **Environment Variables:**
   ```bash
   # Old
   export SERVER_MODE=streamable-http
   export PORT=3000
   
   # New
   export MCP_API_KEY="your-key"
   # Server runs on port 5000 by default
   ```

## 🚨 User Isolation Security

**CRITICAL:** All tool calls now require `user_id` parameter:

- **Memory Isolation**: Each user_id gets completely separate memory storage
- **Data Privacy**: Users cannot access each other's memories
- **Session Context**: Each client maintains isolated session context

**Example with User Isolation:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": {
      "params": {
        "user_input": "Personal note",
        "agent_response": "Understood",
        "user_id": "alice_2024"  // Required for isolation
      }
    }
  },
  "id": 1
}
```

## 📈 Performance and Scaling

### **Resource Requirements**
- **Minimum**: 512MB RAM, 1 CPU core
- **Recommended**: 2GB RAM, 2 CPU cores
- **Storage**: Depends on memory database size per user

### **Concurrent Clients**
- **Supported**: Multiple concurrent MCP clients
- **Isolation**: Each client session completely isolated
- **Memory**: Per-user memory instances scale independently

## 🔧 Configuration Reference

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_API_KEY` | auto-generated | API key for authentication |
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_BASE_URL` | none | Custom OpenAI endpoint |

### **Server Configuration**
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `5000`
- **Protocol**: HTTP (use reverse proxy for HTTPS)

Your MemoryOS Pure MCP 2.0 server is now ready for secure, standards-compliant deployment! 🔐🚀