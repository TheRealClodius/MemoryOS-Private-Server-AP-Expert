# MemoryOS MCP Server Security Guide

## 🔐 Security Overview

MemoryOS MCP server includes comprehensive security features to protect your deployment in production environments. This guide covers all security aspects, from authentication to deployment best practices.

## 🛡️ Authentication Methods

### **1. API Key Authentication**

The server supports multiple authentication methods:

#### **Header-Based Authentication**
```bash
curl -X POST http://localhost:3000/mcp \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

#### **Bearer Token Authentication**
```bash
curl -X POST http://localhost:3000/mcp \
  -H "Authorization: Bearer your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

### **2. API Key Management**

#### **Single API Key (Simple Setup)**
```bash
export MCP_API_KEY="your-secure-api-key"
```

#### **Multiple API Keys (Multi-Client)**
```bash
export MCP_API_KEYS="claude:key1,mobile:key2,web:key3"
```

#### **Auto-Generated Keys**
If no API key is provided, the server generates a secure random key:
```bash
python deploy_server.py
# Output: 🔑 Generated API Key: AbCdEf1234567890...
```

## 🚦 Rate Limiting

Protect against abuse with configurable rate limits:

```bash
# Default: 100 requests per hour
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW=3600

# Custom rate limits
export RATE_LIMIT_REQUESTS=50   # 50 requests
export RATE_LIMIT_WINDOW=1800   # per 30 minutes
```

### **Rate Limiting Features**
- **Per-client IP tracking**
- **Sliding window algorithm**
- **Configurable limits**
- **Automatic cleanup of old requests**
- **429 Too Many Requests response**

## 🌐 CORS Configuration

Control cross-origin access:

```bash
# Enable CORS (default)
export ENABLE_CORS=true
export ALLOWED_ORIGINS="https://claude.ai,https://yourapp.com"

# Disable CORS for strict security
export ENABLE_CORS=false
```

## 🔒 Security Headers

Automatic security headers for all responses:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Strict-Transport-Security` (when HTTPS is enabled)
- `Server: MemoryOS-MCP`

## 🔐 HTTPS Enforcement

For production deployments:

```bash
export REQUIRE_HTTPS=true
export TRUSTED_HOSTS="yourdomain.com,api.yourdomain.com"
```

## 📊 Session Management

### **Session Isolation**
Each API key gets its own session context:
- **Independent memory storage**
- **Separate conversation history**
- **Isolated user profiles**
- **Automatic session cleanup**

### **Session Monitoring**
```bash
# List active sessions
curl -H "X-API-Key: your-key" http://localhost:3000/admin/sessions

# Response
{
  "active_sessions": 3,
  "sessions": ["key1-abc123", "key2-def456", "key3-ghi789"]
}
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

## 📈 Monitoring and Logging

### **Health Checks**

```bash
# Public health check (no auth required)
curl http://localhost:3000/health

# Authenticated admin stats
curl -H "X-API-Key: your-key" http://localhost:3000/admin/stats
```

### **Security Metrics**
The server tracks:
- **Failed authentication attempts**
- **Rate limit violations**
- **Active sessions per API key**
- **Request patterns**

## 🔄 Development Mode

For development, you can disable authentication:

```bash
export DISABLE_AUTH=true
export SERVER_MODE=streamable-http
python deploy_server.py
```

**⚠️ Warning**: Never use `DISABLE_AUTH=true` in production!

## 📚 Security Configuration Reference

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
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

## 🚨 Security Checklist

### **Before Deployment**
- [ ] Generate strong API keys
- [ ] Configure rate limiting
- [ ] Set up HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set trusted hosts
- [ ] Enable security headers
- [ ] Configure CORS appropriately
- [ ] Set up monitoring/logging

### **Production Environment**
- [ ] Never use `DISABLE_AUTH=true`
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS enforcement
- [ ] Set up reverse proxy
- [ ] Configure rate limits appropriately
- [ ] Monitor authentication failures
- [ ] Rotate API keys regularly
- [ ] Set up security alerts

## ✅ Testing Security

```bash
# Test authentication
curl -X POST http://localhost:3000/mcp \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'

# Test rate limiting (send 101 requests rapidly)
for i in {1..101}; do
  curl -H "X-API-Key: your-key" http://localhost:3000/mcp &
done
wait
# Should get 429 Too Many Requests after 100 requests

# Test invalid key
curl -X POST http://localhost:3000/mcp \
  -H "X-API-Key: invalid-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
# Should get 401 Unauthorized
```

## 🔍 Security Audit

### **Common Vulnerabilities Addressed**
- **Authentication bypass** - Multi-layer API key validation
- **Rate limiting bypass** - IP-based tracking with sliding window
- **CORS misconfiguration** - Configurable origin restrictions
- **Information disclosure** - Secure error messages
- **Session hijacking** - Unique session IDs per API key
- **Injection attacks** - Input validation and sanitization

### **Security Headers Implemented**
- **X-Content-Type-Options** - Prevents MIME type sniffing
- **X-Frame-Options** - Prevents clickjacking
- **X-XSS-Protection** - Enables XSS filtering
- **Referrer-Policy** - Controls referrer information
- **Strict-Transport-Security** - Forces HTTPS

## 📞 Security Support

For security-related questions or to report vulnerabilities:
- Create an issue on the GitHub repository
- Label it with `security`
- Do not include sensitive information in public issues

Your MemoryOS MCP server is now secure and ready for production! 🔐