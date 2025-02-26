# GitHub Integration

## Initialize Git Repository
To initialize the Git repository and set up the remote origin, run the following commands:

```bash
git init
git remote add origin https://github.com/yourusername/mcp-ai-autonomous.git

git add .
git commit -m "Initial Commit - MCP AI"
git push origin main
```


### Build & Run the Docker Container
```bash
docker build -t mcp-ai-autonomous .
docker run -d -p 8000:8000 mcp-ai-autonomous
```