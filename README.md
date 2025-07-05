# MemoryOS MCP Server

A production-ready Memory Operating System implemented as an MCP (Model Context Protocol) server. Provides persistent memory capabilities for AI agents through a three-tier hierarchical memory architecture with semantic search.

## Overview

MemoryOS enables AI assistants to maintain persistent memory across conversations by storing, organizing, and retrieving conversation history using OpenAI embeddings and FAISS vector search. The system automatically builds user profiles and knowledge bases while providing intelligent memory consolidation.

## Features

- **Three-tier memory architecture**: Short-term, mid-term, and long-term memory systems
- **Semantic memory retrieval**: Vector similarity search using OpenAI embeddings
- **Automatic user profiling**: Builds personality profiles from conversation history
- **Knowledge extraction**: Identifies and stores important information about users and topics
- **MCP protocol compliance**: Works with Claude Desktop and other MCP clients
- **User data isolation**: Each user has separate, secure data storage

## Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Claude Desktop (for MCP client integration)

### Installation

1. **Clone or download the MemoryOS server files**
2. **Install dependencies:**
   ```bash
   pip install mcp openai numpy faiss-cpu pydantic
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
   
   **⚠️ IMPORTANT**: See `SECURITY.md` for secure configuration details.

4. **Test the server:**
   ```bash
   python mcp_server.py
   ```
   The server should start and display initialization messages.

## Step-by-Step Integration Guide

### Step 1: Get Your OpenAI API Key

1. Visit [platform.openai.com](https://platform.openai.com)
2. Create an account or sign in
3. Navigate to "API Keys" section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)
6. Ensure your account has sufficient credits/quota

### Step 2: Configure MemoryOS

**⚠️ SECURITY NOTICE**: Never put API keys directly in configuration files that might be committed to version control. Use the secure configuration method below.

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your_actual_api_key_here"
```

**Option B: Local Configuration File**
1. Copy the configuration template:
```bash
cp config.template.json config.json
```

2. Add your OpenAI API key to the copied `config.json`:
```bash
# Edit config.json and add: "openai_api_key": "your_actual_api_key_here"
```

The configuration template includes all necessary settings. See `SECURITY.md` for detailed secure configuration guidance.

**Configuration Options:**
- `user_id`: Unique identifier for the user (creates separate data directory)
- `data_storage_path`: Where to store memory data files
- `short_term_capacity`: Number of recent conversations to keep in immediate memory
- `mid_term_capacity`: Maximum consolidated memory segments
- `long_term_knowledge_capacity`: Maximum knowledge entries per category

### Step 3: Test Server Functionality

Run the test script to verify everything works:

```bash
python test_full_functionality.py
```

Expected output:
```
✅ Embedding Generation: Working
✅ LLM Integration: Working  
✅ Complete Memory System: Working
🎉 SUCCESS: All functionality operational!
```

### Step 4: Configure Claude Desktop

1. **Locate Claude Desktop configuration file:**
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. **Edit the configuration file** to add MemoryOS server:

```json
{
  "mcpServers": {
    "memoryos": {
      "command": "python",
      "args": ["/full/path/to/your/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Important**: Use the full absolute path to `mcp_server.py`

Example paths:
- **macOS/Linux**: `/Users/yourname/memoryos/mcp_server.py`
- **Windows**: `C:\Users\yourname\memoryos\mcp_server.py`

3. **Save and restart Claude Desktop**

### Step 5: Verify Integration

1. **Open Claude Desktop**
2. **Look for the hammer icon** (🔨) in the interface - this indicates MCP tools are available
3. **Start a conversation** and the MemoryOS tools should be automatically available

## Using MemoryOS

Once integrated with Claude Desktop, MemoryOS works automatically in the background. The system provides three main capabilities:

### Automatic Memory Storage
Every conversation is automatically stored and organized:
- Recent conversations stay in short-term memory
- Important interactions get consolidated into mid-term memory
- Frequently accessed information promotes to long-term knowledge

### Intelligent Memory Retrieval
When you ask questions, MemoryOS automatically:
- Searches across all memory layers
- Finds relevant past conversations
- Provides context from previous interactions
- Retrieves related knowledge about you or topics

### User Profile Building
MemoryOS builds and maintains:
- Your personality traits and preferences
- Topics you're interested in
- Your expertise areas
- Conversation patterns and styles

## Available MCP Tools

MemoryOS provides three MCP tools that Claude Desktop can use:

### 1. add_memory
Stores new conversation pairs in the memory system.
- **Input**: User question and assistant response
- **Output**: Success confirmation with timestamp

### 2. retrieve_memory  
Searches memory for relevant information.
- **Input**: Search query and optional parameters
- **Output**: Relevant conversation history and knowledge

### 3. get_user_profile
Retrieves user profile and knowledge summary.
- **Input**: Optional parameters for knowledge inclusion
- **Output**: User personality analysis and knowledge entries

## Troubleshooting

### Server Won't Start

**Error: "OpenAI API key is required"**
- Verify your API key is set as environment variable: `echo $OPENAI_API_KEY`
- Or check it's correctly added to your `config.json` file
- Ensure your OpenAI account has available quota
- See `SECURITY.md` for secure configuration guidance

**Error: "Module not found"**
- Install dependencies: `pip install mcp openai numpy faiss-cpu pydantic`
- Ensure you're in the correct directory

### Claude Desktop Integration Issues

**Tools not appearing:**
- Verify the full path to `mcp_server.py` is correct
- Check Claude Desktop configuration syntax is valid JSON
- Restart Claude Desktop after configuration changes

**Server connection errors:**
- Ensure Python is accessible from command line
- Verify all dependencies are installed
- Check file permissions on the server script

### Memory/Performance Issues

**Slow responses:**
- Large memory datasets can slow retrieval
- Consider reducing capacity limits in configuration
- Monitor OpenAI API rate limits

**Storage growing too large:**
- MemoryOS automatically manages capacity limits
- Adjust `*_capacity` settings in configuration
- Old memories are automatically archived

## Advanced Configuration

### Custom Storage Location

```json
{
  "data_storage_path": "/path/to/secure/storage/memoryos_data"
}
```

### Multiple Users

Each user should have a unique `user_id` and separate data directory:

```json
{
  "user_id": "alice",
  "data_storage_path": "./memoryos_data_alice"
}
```

### Alternative OpenAI Endpoints

For custom OpenAI-compatible endpoints:

```json
{
  "openai_base_url": "https://your-custom-endpoint.com/v1"
}
```

## Data Storage

MemoryOS stores data locally in JSON and NumPy files:

```
memoryos_data/
└── your_user_id/
    ├── short_term/
    │   └── memories.json
    ├── mid_term/
    │   ├── memories.json
    │   └── embeddings.npy
    └── long_term/
        ├── user_profile.json
        ├── user_knowledge.json
        ├── assistant_knowledge.json
        └── embeddings/
```

## Security Considerations

- **API Keys**: Store securely, never commit to version control
- **Data Privacy**: All memory data stays local on your machine
- **User Isolation**: Each user has separate data directories
- **No Data Transmission**: MemoryOS only sends queries to OpenAI, never your stored conversations

## Support

If you encounter issues:

1. **Check the logs**: MemoryOS prints detailed error messages
2. **Verify configuration**: Ensure `config.json` syntax is correct
3. **Test components**: Run individual test scripts to isolate issues
4. **Check OpenAI status**: Verify your API key and account quota

For additional help, refer to `DEPLOYMENT.md` for detailed technical documentation.

## Example Usage

Once set up, you can have conversations like:

**You**: "I'm working on a Python project for data analysis"
**Claude**: *Uses MemoryOS to store this information*

**Later...**

**You**: "What was I working on yesterday?"
**Claude**: *Uses MemoryOS to retrieve: "You mentioned working on a Python project for data analysis"*

The system learns your preferences, remembers your projects, and maintains context across all conversations.