import json
import chromadb
from transformers import T5Tokenizer, T5Model
import torch

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection in ChromaDB
collection = client.get_or_create_collection(name="api_doc")

# Load T5 Tokenizer & Model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5Model.from_pretrained("t5-small")

# Load JSON Data
with open("data/input/api_to_curl_dataset.json", "r") as f:
    json_data = json.load(f)

# Store JSON data in ChromaDB
for entry in json_data:
    api_doc = entry["api_documentation"]
    curl_cmd = entry["curl_command"]
    combined_text = f"{api_doc} || {curl_cmd}"  # Concatenating both fields

    # Convert to Embeddings
    inputs = tokenizer(combined_text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.encoder(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    # Store in ChromaDB
    collection.add(
        ids=[str(entry["id"])],  # Using the provided ID
        embeddings=[embeddings],
        metadatas=[{"api_documentation": api_doc, "curl_command": curl_cmd}]  # Store API & cURL
    )

print("JSON data stored in ChromaDB successfully ✅")
