import sys

from fastapi import FastAPI
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import chromadb

# Initialize FastAPI
app = FastAPI()

# Load Model and Tokenizer
model_path = "models/t5_api_to_curl_v1"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="api_doc")

# API Endpoint to Handle Queries
def process_query(query: str):
    # Convert Query into Embeddings
    query_inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.encoder(**query_inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    # Retrieve Top 3 Relevant Documents from ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    print("Retrieved data from chroma: ", results)

    # Extract Relevant Texts
    retrieved_texts = [f'{entry["api_documentation"]} || {entry["curl_command"]}' for entry in results["metadatas"][0]]

    # Generate Response Using T5 Model
    context = " ".join(retrieved_texts)
    input_text = f"Given the following API documentation examples\n{context}, \ngenerate the correct cURL command for this query:\nQuery: {query}\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    print("Input IDs: ", input_ids)

    # Generate Answer
    output_ids = model.generate(input_ids)
    print("Output IDs: ", output_ids)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Answer: ", answer)

    return answer

# Run: uvicorn src.deploy_model:app --reload
# Get API text from CLI argument
if __name__ == "__main__":
    query = sys.argv[1]  # Read from command-line argument
    generated_curl = process_query(query)
    print(generated_curl)  # Print output so FastAPI can capture it

