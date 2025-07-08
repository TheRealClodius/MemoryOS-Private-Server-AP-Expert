# MemoryOS Remote MCP Server - Client Authentication Setup

## Current Deployment Status

✅ **MCP 2.0 Remote Server Running**
- **URL**: `http://localhost:5000/mcp` (local development)
- **URL**: `https://memory-os-private-server-ac.replit.app/mcp` (production)
- **Transport**: Streamable HTTP (MCP 2.0 compliant)
- **Authentication**: API Key Required

## 🔑 API Key for Client Authentication

**Current API Key**: `77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4`

## ❌ Client Repository Issue Found

The client repository at `https://github.com/TheRealClodius/autopilot-expert-export-replit` is missing API key configuration:

### Current Client Config (Broken):
```python
# config.py - MISSING API KEY
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL",
    "https://memory-os-private-server-ac.replit.app" if os.getenv("REPLIT_DEPLOYMENT")
    else "http://localhost:8001")
```

### Required Client Config (Fixed):
```python
# config.py - ADD THIS
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", 
    "https://memory-os-private-server-ac.replit.app/mcp" if os.getenv("REPLIT_DEPLOYMENT")
    else "http://localhost:5000/mcp")
MCP_API_KEY: str = os.getenv("MCP_API_KEY", "77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4")
```

### Required Environment Variables:
```bash
# Add to client .env file
MCP_API_KEY=77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4
MCP_SERVER_URL=https://memory-os-private-server-ac.replit.app/mcp
```

## 🔧 Client Implementation Fix

Based on MCP 2.0 best practices, the client needs to implement proper authentication:

### FastMCP Client with Authentication:
```python
import asyncio
from fastmcp import Client

async def connect_to_remote_mcp():
    headers = {
        "Authorization": f"Bearer {settings.MCP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    async with Client(settings.MCP_SERVER_URL, headers=headers) as client:
        # Now client calls will work
        tools = await client.list_tools()
        result = await client.call_tool("add_memory", {...})
```

### HTTP Client with Authentication:
```python
import httpx

async def call_mcp_endpoint():
    headers = {
        "Authorization": f"Bearer {settings.MCP_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.MCP_SERVER_URL,
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
            headers=headers
        )
```

## 🚀 Next Steps for Client Fix

1. **Update client config.py** - Add MCP_API_KEY setting
2. **Add environment variable** - Set MCP_API_KEY in deployment
3. **Update MCP client code** - Include Authorization header
4. **Fix endpoint URL** - Add `/mcp` suffix to URL
5. **Test authentication** - Verify API key is accepted

## 🔐 Security Notes

- The API key is currently logged in development mode
- For production, remove API key logging
- Each deployment should have unique API keys
- Consider implementing OAuth 2.1 for enterprise deployments

## ✅ Verification Commands

Test the authentication from client repository:

```bash
# Test with curl (should work)
curl -X POST https://remote-mcp-server-andreiclodius.replit.app/mcp \
  -H "Authorization: Bearer 77gOCTIGuZLslr-vIk8uTsWF0PZmMgyU8RxMKn_VZd4" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}'

# Test without auth (should fail with 401)
curl -X POST https://remote-mcp-server-andreiclodius.replit.app/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}'
```