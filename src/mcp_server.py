from fastapi import FastAPI
import subprocess
import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configure logging
logging.basicConfig(filename="logs/ai_logs.log", level=logging.INFO)

app = FastAPI()

# Load Model & Tokenizer (If already trained)
model_path = "models/t5_api_to_curl"
try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
except:
    model = None
    tokenizer = None


# Endpoint to generate dataset
@app.get("/generate_dataset/")
async def generate_dataset():
    subprocess.run(["python", "src/generate_dataset.py"])
    return {"status": "Dataset Generated"}


# Endpoint to preprocess dataset
@app.get("/preprocess_data/")
async def preprocess_data():
    subprocess.run(["python", "src/preprocess_data.py"])
    return {"status": "Dataset Preprocessed"}


# Endpoint to train model
@app.get("/train_model/")
async def train_model():
    subprocess.run(["python", "src/train_model.py"])
    return {"status": "Model Training Started"}


# Endpoint to fine-tune model
@app.get("/auto_finetune/")
async def auto_finetune():
    subprocess.run(["python", "src/finetune_model.py"])
    return {"status": "Auto Fine-Tuning Started"}


# Endpoint for inference (Convert API Docs to cURL)
@app.post("/generate_curl/")
async def generate_curl(api_text: str):
    subprocess.run(["python", "src/deploy_model.py"])
    return {"status": "Generation Started"}

# Run the server using `uvicorn mcp_server:app --reload`
