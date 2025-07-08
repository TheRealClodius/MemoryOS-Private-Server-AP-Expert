# Use the official Python lightweight image
FROM python:3.13-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the project into /app
COPY . /app
WORKDIR /app

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN uv sync

# Expose the port
EXPOSE $PORT

# Run the MemoryOS Remote MCP server
CMD ["uv", "run", "mcp_remote_server.py"]