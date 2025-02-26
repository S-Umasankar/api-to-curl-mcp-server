#!/bin/bash
#echo "Installing libraries"
#pip install -r requirements.txt
echo "🚀 Starting MCP Server..."
uvicorn src.mcp_server:app --reload
python src/ai_autonomous_dev.py
