# MemoryOS User Management Guide

## Overview

MemoryOS now supports dynamic user management with automatic user ID generation and isolated data storage. Each user gets their own memory space with separate short-term, mid-term, and long-term memory systems.

## User ID Management

### Automatic User ID Generation
- If no user ID is specified, MemoryOS automatically generates a unique ID: `user_[8-character-uuid]`
- Example: `user_e4e473a0`, `user_b2f91c85`

### Manual User ID Assignment
Set a custom user ID using environment variables:
```bash
export MEMORYOS_USER_ID="alice_2024"
```

Or specify in the configuration file:
```json
{
  "user_id": "custom_user_id"
}
```

## Data Isolation

### User-Specific Storage
Each user gets their own data directory:
- Base path: `./memoryos_data/`
- User data: `./memoryos_data/{user_id}/`
- Example: `./memoryos_data/alice_123/`

### Directory Structure
```
memoryos_data/
├── user_e4e473a0/
│   ├── short_term_memory.json
│   ├── mid_term_memory.json
│   ├── user_profile.json
│   └── embeddings/
└── alice_123/
    ├── short_term_memory.json
    ├── mid_term_memory.json
    ├── user_profile.json
    └── embeddings/
```

## Environment Variables

Configure user settings with environment variables:

```bash
# User identification
export MEMORYOS_USER_ID="your_user_id"
export MEMORYOS_ASSISTANT_ID="your_assistant_name"

# Data storage
export MEMORYOS_DATA_PATH="./custom_data_path"

# OpenAI configuration
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Memory configuration
export MEMORYOS_SHORT_TERM_CAPACITY="10"
export MEMORYOS_MID_TERM_CAPACITY="2000"
export MEMORYOS_KNOWLEDGE_CAPACITY="100"
```

## API Endpoints for User Management

### Get Current User Information
```bash
GET /api/user_info
```

Response:
```json
{
  "status": "success",
  "user_id": "user_e4e473a0",
  "assistant_id": "mcp_assistant",
  "data_path": "./memoryos_data/user_e4e473a0",
  "memory_stats": { ... },
  "timestamp": "2025-07-06T11:39:42.190787"
}
```

### Create New User
```bash
POST /api/create_user
Content-Type: application/json

{
  "user_id": "alice_123",
  "assistant_id": "alice_assistant"
}
```

Response:
```json
{
  "status": "success",
  "message": "User created successfully",
  "user_id": "alice_123",
  "assistant_id": "alice_assistant", 
  "data_path": "./memoryos_data/alice_123",
  "timestamp": "2025-07-06T11:39:46.304253"
}
```

### Get User Profile
```bash
GET /api/user_profile
```

Response:
```json
{
  "status": "success",
  "timestamp": "2025-07-06T11:39:42.190787",
  "user_id": "user_e4e473a0",
  "user_profile": "None"
}
```

## Multi-User Deployment

### Method 1: Environment Variables
Deploy multiple instances with different environment variables:

```bash
# Instance 1
export MEMORYOS_USER_ID="user1"
export MEMORYOS_DATA_PATH="./data"
python main.py

# Instance 2  
export MEMORYOS_USER_ID="user2"
export MEMORYOS_DATA_PATH="./data"
python main.py
```

### Method 2: Configuration Files
Create separate configuration files:

**config_user1.json:**
```json
{
  "user_id": "user1",
  "data_storage_path": "./data"
}
```

**config_user2.json:**
```json
{
  "user_id": "user2", 
  "data_storage_path": "./data"
}
```

### Method 3: Dynamic User Creation
Use the API to create users on-demand:

```bash
# Create users via API
curl -X POST "https://your-service.replit.app/api/create_user" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "team_member_1"}'

curl -X POST "https://your-service.replit.app/api/create_user" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "team_member_2"}'
```

## Security Considerations

### Data Isolation
- Each user's data is completely isolated
- No cross-user data access possible
- User directories are automatically created with proper permissions

### User ID Validation
- User IDs are validated and sanitized
- Invalid characters are rejected
- Directory traversal attacks prevented

### Environment Security
- OpenAI API key should be set via environment variables
- Never hardcode API keys in configuration files
- Use Replit secrets for production deployments

## Best Practices

### User ID Naming
- Use descriptive, unique user IDs: `alice_2024`, `team_lead`, `customer_123`
- Avoid special characters except underscores and hyphens
- Keep IDs under 50 characters for filesystem compatibility

### Data Management
- Regularly backup user data directories
- Monitor disk space usage as each user accumulates memory data
- Consider implementing data retention policies for long-term deployments

### Performance
- Each user maintains separate memory indexes
- Memory retrieval is user-specific and does not cross boundaries
- Consider memory limits for high-user-count deployments

## Migration from Single User

If upgrading from a single-user deployment:

1. **Backup existing data:**
   ```bash
   cp -r ./memoryos_data ./memoryos_data_backup
   ```

2. **Create user directory:**
   ```bash
   mkdir -p ./memoryos_data/your_user_id
   mv ./memoryos_data/*.json ./memoryos_data/your_user_id/
   ```

3. **Update configuration:**
   ```bash
   export MEMORYOS_USER_ID="your_user_id"
   ```

4. **Restart service**

## Troubleshooting

### User Not Found
If you get user-related errors:
- Check `MEMORYOS_USER_ID` environment variable
- Verify user directory exists in data path
- Check file permissions on data directory

### Data Path Issues
If you encounter data path errors:
- Ensure `MEMORYOS_DATA_PATH` is writable
- Check disk space availability
- Verify directory permissions

### Memory Isolation Issues
If users see each other's data:
- Restart the service to reload configuration
- Verify environment variables are set correctly
- Check that user IDs are unique and properly configured