#!/bin/bash
echo "🚀 Starting MCP Server..."
uvicorn src.mcp_server:app --reload
