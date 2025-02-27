#!/bin/bash
#echo "Installing libraries"
#rm -rf venv
#python -m venv venv
#source venv/bin/activate  # macOS/Linux
#venv\Scripts\activate  # Windows
#pip install -r requirements.txt
echo "🚀 Starting MCP Server..."
uvicorn src.mcp_server:app --reload
python src/ai_autonomous_dev.py
