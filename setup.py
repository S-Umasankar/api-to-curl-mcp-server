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
    author="Umasankar Srinivasan",
    description="MCP SDK for API-to-cURL Model Automation",
    license=""
)
