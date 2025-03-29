from fastapi import FastAPI
import subprocess
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configure logging
logging.basicConfig(filename="logs/ai_logs.log", level=logging.INFO)

app = FastAPI()

# Load Model & Tokenizer (If already trained)
model_path = "models/t5_api_to_curl_v1"
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

# Endpoint to setup RAG workflow
@app.get("/setup_rag/")
async def setup_rag():
    subprocess.run(["python", "src/vector_db.py"])
    return {"status": "Vector DB run completed"}

# Endpoint to execute RAG retrieval
@app.post("/rag_retrieval")
async def rag_retrieval(query: str):
    result = subprocess.run(
        ["python", "src/rag_retrieval.py", query])
    return {"RAG output": result}  # Return generated cURL

# Evaluate to train model
@app.get("/evaluate_model/")
async def train_model():
    subprocess.run(["python", "src/evaluate_model.py"])
    return {"status": "Model Training Started"}

# Testing the model
@app.get("/test_training/")
async def test_model():
    subprocess.run(["python", "tests/test_training.py"])
    return {"status": "Model Testing Started"}

# Testing the inference
@app.get("/test_inference/")
async def test_inference():
    subprocess.run(["python", "tests/test_inference.py"])
    return {"status": "Inference Testing Started"}

# Endpoint to fine-tune model
@app.get("/auto_finetune/")
async def auto_finetune():
    subprocess.run(["python", "src/finetune_model.py"])
    return {"status": "Auto Fine-Tuning Started"}

@app.post("/generate_curl/")
async def generate_curl(api_text: str):
    try:
        # Run generate_curl.py and capture the output
        result = subprocess.run(
            ["python", "src/generate_curl.py", api_text],  # Pass input as argument
            capture_output=True, text=True, check=True
        )

        return {"curl_command": result.stdout.strip()}  # Return generated cURL

    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to generate cURL: {e.stderr}"}

# Run the server using `uvicorn mcp_server:app --reload`
