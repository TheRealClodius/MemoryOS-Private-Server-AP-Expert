# MemoryOS Remote MCP Server Deployment Guide

## Overview

MemoryOS now supports **remote MCP deployment** using the **StreamableHTTP transport**, allowing multiple applications to connect to a single MemoryOS instance over HTTP with **comprehensive security features**.

## Architecture

```
┌─────────────────────┐     HTTP/JSON-RPC     ┌──────────────────────┐
│   MCP Client 1      │ ←------------------→ │                      │
│   (Claude Desktop)  │                      │                      │
└─────────────────────┘                      │                      │
                                             │   MemoryOS MCP       │
┌─────────────────────┐     HTTP/JSON-RPC     │   Server (Secure)    │
│   MCP Client 2      │ ←------------------→ │   (StreamableHTTP)   │
│   (Custom App)      │                      │                      │
└─────────────────────┘                      │   🔐 Authentication  │
                                             │   🚦 Rate Limiting   │
┌─────────────────────┐     HTTP/JSON-RPC     │   🛡️  Security       │
│   MCP Client N      │ ←------------------→ │   📊 Session Mgmt    │
│   (Mobile App)      │                      │                      │
└─────────────────────┘                      └──────────────────────┘
```

## 🔐 Security Features

### **1. API Key Authentication**

The server supports **multiple authentication methods**:

#### **Method 1: Header-Based Authentication**
```bash
curl -X POST http://localhost:3000/mcp \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

#### **Method 2: Bearer Token Authentication**
```bash
curl -X POST http://localhost:3000/mcp \
  -H "Authorization: Bearer your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

### **2. API Key Management**

#### **Option A: Environment Variables**
```bash
# Single API key for all clients
export MCP_API_KEY="your-secure-api-key"

# Multiple API keys for different clients
export MCP_API_KEYS="app1:key1,app2:key2,app3:key3"
```

#### **Option B: Auto-Generated Keys**
If no API key is provided, the server generates a secure random key:
```bash
python deploy_server.py
# Output: 🔑 Generated API Key: AbCdEf1234567890...
```

### **3. Rate Limiting**

Protect against abuse with configurable rate limits:

```bash
# Default: 100 requests per hour
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW=3600

# Custom rate limits
export RATE_LIMIT_REQUESTS=50   # 50 requests
export RATE_LIMIT_WINDOW=1800   # per 30 minutes
```

### **4. CORS Configuration**

Control cross-origin access:

```bash
# Enable CORS (default)
export ENABLE_CORS=true
export ALLOWED_ORIGINS="https://claude.ai,https://yourapp.com"

# Disable CORS for strict security
export ENABLE_CORS=false
```

### **5. Security Headers**

Automatic security headers for all responses:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Strict-Transport-Security` (when HTTPS is enabled)

### **6. HTTPS Enforcement**

For production deployments:

```bash
export REQUIRE_HTTPS=true
export TRUSTED_HOSTS="yourdomain.com,api.yourdomain.com"
```

## 🚀 Deployment Options

### **Option 1: Direct Python Deployment**

```bash
# Basic deployment
export MCP_API_KEY="your-secure-key"
export SERVER_MODE="streamable-http"
export PORT=3000
python deploy_server.py
```

### **Option 2: Docker Deployment**

```bash
# Build and run with Docker
docker build -t memoryos-mcp .
docker run -p 3000:3000 \
  -e MCP_API_KEY="your-secure-key" \
  -e SERVER_MODE="streamable-http" \
  -v $(pwd)/memoryos_data:/app/memoryos_data \
  memoryos-mcp
```

### **Option 3: Docker Compose with Security**

```yaml
# docker-compose.yml
version: '3.8'

services:
  memoryos-mcp:
    build: .
    ports:
      - "3000:3000"
    environment:
      - SERVER_MODE=streamable-http
      - MCP_API_KEYS=claude:secure-key-1,mobile:secure-key-2
      - RATE_LIMIT_REQUESTS=200
      - RATE_LIMIT_WINDOW=3600
      - ENABLE_CORS=true
      - ALLOWED_ORIGINS=https://claude.ai,https://yourapp.com
    volumes:
      - ./memoryos_data:/app/memoryos_data
```

## 🔧 Client Configuration

### **Claude Desktop Configuration**

Create or update `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memoryos-remote": {
      "type": "streamable-http",
      "url": "http://localhost:3000/mcp",
      "headers": {
        "X-API-Key": "your-secure-api-key"
      }
    }
  }
}
```

### **Custom Python Client**

```python
import httpx
import json

class MemoryOSClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def add_memory(self, user_input: str, agent_response: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp",
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "add_memory",
                        "arguments": {
                            "user_input": user_input,
                            "agent_response": agent_response
                        }
                    },
                    "id": 1
                }
            )
            return response.json()

# Usage
client = MemoryOSClient("http://localhost:3000", "your-secure-api-key")
result = await client.add_memory("Hello", "Hi there!")
```

## 🛡️ Production Security Recommendations

### **1. Use Strong API Keys**

Generate cryptographically secure API keys:

```bash
# Generate secure keys
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### **2. Enable HTTPS**

Use a reverse proxy with SSL termination:

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.key;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### **3. Firewall Configuration**

```bash
# UFW example
sudo ufw allow 443/tcp
sudo ufw allow 80/tcp
sudo ufw deny 3000/tcp  # Block direct access
```

### **4. Environment Variables**

Use secure environment variable management:

```bash
# .env file (never commit to git)
MCP_API_KEYS=production-app:very-secure-key-here
RATE_LIMIT_REQUESTS=50
OPENAI_API_KEY=your-openai-key
REQUIRE_HTTPS=true
TRUSTED_HOSTS=yourdomain.com
```

## 📊 Monitoring and Admin

### **Health Checks**

```bash
# Public health check (no auth required)
curl http://localhost:3000/health

# Authenticated admin stats
curl -H "X-API-Key: your-key" http://localhost:3000/admin/stats
```

### **Session Management**

```bash
# List active sessions
curl -H "X-API-Key: your-key" http://localhost:3000/admin/sessions

# Response
{
  "active_sessions": 3,
  "sessions": ["key1-abc123", "key2-def456", "key3-ghi789"]
}
```

## 🔄 Development Mode

For development, you can disable authentication:

```bash
export DISABLE_AUTH=true
export SERVER_MODE=streamable-http
python deploy_server.py
```

**⚠️ Warning**: Never use `DISABLE_AUTH=true` in production!

## 📚 Configuration Reference

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_MODE` | `streamable-http` | Server transport mode |
| `PORT` | `3000` | Server port |
| `MCP_API_KEY` | auto-generated | Single API key |
| `MCP_API_KEYS` | none | Multiple API keys (name:key,name:key) |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per window |
| `RATE_LIMIT_WINDOW` | `3600` | Rate limit window (seconds) |
| `ENABLE_CORS` | `true` | Enable CORS |
| `ALLOWED_ORIGINS` | `*` | Allowed CORS origins |
| `TRUSTED_HOSTS` | `*` | Trusted host patterns |
| `REQUIRE_HTTPS` | `false` | Require HTTPS |
| `DISABLE_AUTH` | `false` | Disable auth (dev only) |

### **API Endpoints**

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/` | No | Health check |
| `GET` | `/health` | No | Detailed health |
| `POST` | `/mcp` | Yes | MCP JSON-RPC |
| `GET` | `/mcp` | Yes | MCP SSE (future) |
| `DELETE` | `/mcp` | Yes | Disconnect session |
| `GET` | `/admin/sessions` | Yes | List sessions |
| `GET` | `/admin/stats` | Yes | Server statistics |

## 🔐 User Isolation Security

**CRITICAL:** MemoryOS MCP server now provides complete user isolation:

- **Per-User Memory**: Each API key gets its own isolated memory instance
- **File System Isolation**: Users get separate data directories
- **No Data Leakage**: Alice's conversations never visible to Bob
- **Session Isolation**: Each client session is completely isolated

### **Verification**
```bash
# Test that user isolation is working
python3 test_user_isolation.py
# ✅ USER ISOLATION FIX VERIFIED SUCCESSFULLY!
```

## 🚨 Security Considerations

1. **Never commit API keys** to version control
2. **Use HTTPS in production** environments
3. **Implement proper firewall rules**
4. **Monitor rate limits** and failed authentication attempts
5. **Rotate API keys regularly**
6. **Use strong, unique keys** for each client
7. **Enable logging** for security auditing
8. **User isolation is enabled by default** (no additional configuration needed)

## 🎯 Multi-Client Examples

### **Multiple Apps with Different Keys**

```bash
# Set up different keys for different apps
export MCP_API_KEYS="claude:key1,mobile:key2,web:key3"

# Each app uses its own key
# Claude Desktop uses "key1"
# Mobile app uses "key2"  
# Web app uses "key3"
```

### **Session Isolation**

Each API key gets its own session context:
- **Independent memory storage**
- **Separate conversation history**
- **Isolated user profiles**

This allows multiple applications to use the same MemoryOS instance without interfering with each other.

## ✅ Testing the Secure Setup

```bash
# Test authentication
curl -X POST http://localhost:3000/mcp \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'

# Test rate limiting (send 101 requests rapidly)
for i in {1..101}; do
  curl -H "X-API-Key: your-key" http://localhost:3000/mcp
done
# Should get 429 Too Many Requests after 100 requests

# Test invalid key
curl -X POST http://localhost:3000/mcp \
  -H "X-API-Key: invalid-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
# Should get 401 Unauthorized
```

Your MemoryOS MCP server is now ready for secure, multi-client deployment! 🔐🚀

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export SERVER_MODE=streamable-http
export PORT=3000
export OPENAI_API_KEY=your-openai-api-key
```

### 3. Run the Server

```bash
python deploy_server.py
```

Your MemoryOS MCP server is now running at `http://localhost:3000`

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t memoryos-mcp .

# Run the container
docker run -p 3000:3000 -e OPENAI_API_KEY=your-key memoryos-mcp
```

### Using Docker Compose

```bash
# Start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

## Client Configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memoryos-remote": {
      "type": "streamable-http",
      "url": "http://localhost:3000/mcp"
    }
  }
}
```

### Custom MCP Clients

Connect to the server using:

- **Endpoint**: `http://your-server:3000/mcp`
- **Method**: POST
- **Content-Type**: application/json
- **Protocol**: JSON-RPC 2.0

Example request:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

## Production Deployment

### Environment Variables

```bash
# Server Configuration
SERVER_MODE=streamable-http
PORT=3000
HOST=0.0.0.0

# MemoryOS Configuration
OPENAI_API_KEY=your-openai-api-key

# Optional: Custom data directory
MEMORYOS_DATA_DIR=/app/data
```

### Cloud Platforms

#### AWS/GCP/Azure

1. Deploy as a container service
2. Set environment variables
3. Expose port 3000
4. Configure load balancer if needed

#### Heroku

```bash
# Create app
heroku create memoryos-mcp

# Set environment variables
heroku config:set SERVER_MODE=streamable-http
heroku config:set OPENAI_API_KEY=your-key

# Deploy
git push heroku main
```

#### Railway

```bash
# Deploy with one command
railway deploy
```

### Reverse Proxy (Production)

Example nginx configuration:

```nginx
upstream memoryos_mcp {
    server localhost:3000;
}

server {
    listen 80;
    server_name your-domain.com;

    location /mcp {
        proxy_pass http://memoryos_mcp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## API Endpoints

### MCP Protocol Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp` | POST | Main MCP JSON-RPC endpoint |
| `/mcp` | GET | SSE endpoint (future) |
| `/mcp` | DELETE | Session termination |

### Health Check Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic health check |
| `/health` | GET | Detailed health and statistics |

## Available Tools

### add_memory
Add new memories to MemoryOS.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": {
      "user_input": "What is the weather like?",
      "agent_response": "The weather is sunny and 75°F",
      "memory_type": "conversation",
      "tags": ["weather", "question"]
    }
  },
  "id": 1
}
```

### retrieve_memory
Retrieve relevant memories from MemoryOS.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "retrieve_memory",
    "arguments": {
      "query": "weather information",
      "max_results": 5
    }
  },
  "id": 2
}
```

### get_user_profile
Get comprehensive user profile information.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_user_profile",
    "arguments": {
      "user_id": "default"
    }
  },
  "id": 3
}
```

## Scaling and Performance

### Multiple Clients

- **Concurrent connections**: Supported via FastAPI async handling
- **Session isolation**: Each client gets independent session
- **Load balancing**: Deploy behind load balancer for high availability

### Resource Requirements

- **Minimum**: 512MB RAM, 1 CPU core
- **Recommended**: 2GB RAM, 2 CPU cores
- **Storage**: Depends on memory database size

### Monitoring

Health check endpoints provide:
- Server status
- Memory statistics
- Active session count
- Performance metrics

## Security Considerations

### Current State
- **No authentication**: Open access to all clients
- **No encryption**: HTTP only (add HTTPS via reverse proxy)
- **No rate limiting**: Implement via reverse proxy

### Production Recommendations
1. **Use HTTPS** via reverse proxy (nginx, cloudflare, etc.)
2. **Add authentication** via reverse proxy or API gateway
3. **Implement rate limiting** to prevent abuse
4. **Use firewall rules** to restrict access
5. **Monitor logs** for suspicious activity

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 3000
   lsof -i :3000
   # Kill the process or use different port
   export PORT=3001
   ```

2. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **OpenAI API key issues**
   ```bash
   # Check if key is set
   echo $OPENAI_API_KEY
   # Set the key
   export OPENAI_API_KEY=your-key-here
   ```

4. **Memory persistence issues**
   ```bash
   # Check data directory permissions
   ls -la memoryos_data/
   # Create if missing
   mkdir -p memoryos_data
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python deploy_server.py
```

## Migration from stdio MCP

If you're migrating from stdio MCP:

1. **Backup existing data**:
   ```bash
   cp -r memoryos_data memoryos_data.backup
   ```

2. **Update client configuration**:
   - Change from `command` to `type: "streamable-http"`
   - Add `url` instead of command path

3. **Test the migration**:
   ```bash
   # Test old stdio mode
   SERVER_MODE=stdio python mcp_server.py
   
   # Test new remote mode
   SERVER_MODE=streamable-http python deploy_server.py
   ```

## Examples

Check the `remote_mcp_config.json` file for:
- Complete configuration examples
- Client setup instructions
- API usage examples
- Deployment patterns

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Verify client configuration
4. Test with curl/Postman first

## Development

To contribute or modify:

1. **Local development**:
   ```bash
   # Install in development mode
   pip install -e .
   
   # Run tests
   python -m pytest tests/
   ```

2. **Code changes**:
   - Server logic: `mcp_server.py`
   - Deployment: `deploy_server.py`
   - Docker: `Dockerfile`, `docker-compose.yml`

3. **Testing**:
   ```bash
   # Test endpoints
   curl -X POST http://localhost:3000/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
   ``` 