from fastapi import FastAPI
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys

app = FastAPI()

@app.post("/generate_curl/")
def generate_curl(api_text: str):
    # Load model & tokenizer
    model = T5ForConditionalGeneration.from_pretrained("models/t5_api_to_curl").to("cpu")
    tokenizer = T5Tokenizer.from_pretrained("models/t5_api_to_curl")

    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    model.eval()

    # Tokenize input API doc
    input_tokens = tokenizer(api_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate output
    output_ids = model.generate(input_tokens["input_ids"], max_length=256, num_beams=10, early_stopping=True)

    # Decode the generated output
    generated_curl = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    with open("data/output.txt", "w") as f:
        sys.stdout = f  # Redirects all print output to the file

        # Print raw output tokens
        print("Generated Output IDs:", output_ids)
        print("Generated cURL Command:", generated_curl)
    sys.stdout = sys.__stdout__

    return generated_curl

# Run: uvicorn src.deploy_model:app --reload