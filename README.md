# 🚀 MCP-AI: Self-Learning API-to-cURL Model

This project builds an **autonomous AI system** to convert API documentation into cURL commands.

## 📌 Features:
✅ **Automated Dataset Generation**  
✅ **Self-Improving Model** with Reinforcement Learning  
✅ **MCP Server for API-based Execution**  
✅ **Continuous Deployment with GitHub Actions**  

---

## 🚀 Quick Start:

### 1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

### 2️⃣ Start MCP Server:  
```bash
bash scripts/start_mcp.sh
```

### 3️⃣ Start AI Automation:  
```bash
python src/ai_autonomous_dev.py
```

### 4️⃣ Test System:  
```bash
pytest tests/
```

---

## 📜 `setup.py` (For Packaging SDK)
```python
from setuptools import setup, find_packages

setup(
    name="mcp_sdk",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "transformers",
        "sacrebleu",
        "requests",
        "pytest",
        "gitpython",
    ],
    author="Your Name",
    description="MCP SDK for API-to-cURL Model Automation",
    license="MIT"
)
```

---

## ✅ Final Steps

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start MCP Server
```bash
bash scripts/start_mcp.sh
```

### 3️⃣ Run AI Automation
```bash
python src/ai_autonomous_dev.py
```

### 4️⃣ Test System
```bash
pytest tests/
```

### Fix uvicorn: command not found
The error indicates that uvicorn is not installed or not in the system path.

### ✅ Solution 1: Install Uvicorn
```bash
pip install uvicorn
```
### ✅ Solution 2: Ensure Virtual Environment is Activated
```bash
source /Users/umasankars/PycharmProjects/CapstoneMCPserver/venv/bin/activate
pip install -r requirements.txt
```
### ✅ Solution 3: Explicitly Call Python for Uvicorn
Modify scripts/start_mcp.sh to:

```bash

#!/bin/bash
echo "🚀 Starting MCP Server..."
/Users/umasankars/PycharmProjects/CapstoneMCPserver/venv/bin/python -m uvicorn src.mcp_server:app --reload
```
### Final Steps
After applying the fixes, restart everything:

```bash

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
bash scripts/start_mcp.sh
```
🚀 **Now the system is fully organized and self-learning!**  🎯

