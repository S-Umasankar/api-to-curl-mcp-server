from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys

def generate_curl(api_text: str):
    # Load model & tokenizer
    model = T5ForConditionalGeneration.from_pretrained("models/t5_api_to_curl").to("cpu")
    tokenizer = T5Tokenizer.from_pretrained("models/t5_api_to_curl")

    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    with open("data/output/output.txt", "w") as f:
        sys.stdout = f  # Redirects all print output to the file

        print("Evaluation starts...")
        model.eval()

        # Tokenize input API doc
        input_tokens = tokenizer(api_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate output
        output_ids = model.generate(input_tokens["input_ids"], max_length=256, num_beams=10, early_stopping=True)

        print("Generating the curl command...")
        # Decode the generated output
        generated_curl = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Generated output: ", generated_curl)

        # Print raw output tokens
        print("Generated Output IDs:", output_ids)
        print("Generated cURL Command:", generated_curl)
    sys.stdout = sys.__stdout__

    return generated_curl

# Run: uvicorn src.deploy_model:app --reload
# Get API text from CLI argument
if __name__ == "__main__":
    api_text = sys.argv[1]  # Read from command-line argument
    generated_curl = generate_curl(api_text)
    print(generated_curl)  # Print output so FastAPI can capture it