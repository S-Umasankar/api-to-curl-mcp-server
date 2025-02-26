from fastapi import FastAPI
import subprocess
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

# Load Model & Tokenizer (If already trained)
model_path = "t5_api_to_curl"
try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
except:
    model = None
    tokenizer = None


# Endpoint to generate dataset
@app.get("/generate_dataset/")
async def generate_dataset():
    subprocess.run(["python", "generate_dataset.py"])
    return {"status": "Dataset Generated"}


# Endpoint to preprocess dataset
@app.get("/preprocess_data/")
async def preprocess_data():
    subprocess.run(["python", "preprocess_data.py"])
    return {"status": "Dataset Preprocessed"}


# Endpoint to train model
@app.get("/train_model/")
async def train_model():
    subprocess.run(["python", "train_model.py"])
    return {"status": "Model Training Started"}


# Endpoint to fine-tune model
@app.get("/auto_finetune/")
async def auto_finetune():
    subprocess.run(["python", "finetune_model.py"])
    return {"status": "Auto Fine-Tuning Started"}


# Endpoint for inference (Convert API Docs to cURL)
@app.post("/generate_curl/")
async def generate_curl(api_text: str):
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    input_tokens = tokenizer(api_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_ids = model.generate(input_tokens["input_ids"])
    generated_curl = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"curl_command": generated_curl}

# Run the server using `uvicorn mcp_server:app --reload`
