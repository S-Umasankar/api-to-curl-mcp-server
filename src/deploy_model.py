from fastapi import FastAPI
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("models/t5_api_to_curl").to("cpu")
tokenizer = T5Tokenizer.from_pretrained("models/t5_api_to_curl")

app = FastAPI()

@app.post("/generate_curl/")
def generate_curl(api_text: str):
    input_tokens = tokenizer(api_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_ids = model.generate(input_tokens["input_ids"])
    generated_curl = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"curl_command": generated_curl}

# Run: uvicorn src.deploy_model:app --reload
