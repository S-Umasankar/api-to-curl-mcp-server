#!/bin/bash
while true
do
    echo "🔄 Starting AI Auto-Execution..."
    python src/ai_autonomous_dev.py
    echo "⚠️ AI process crashed. Restarting..."
    sleep 5
done
