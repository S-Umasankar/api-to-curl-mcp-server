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
for i, entry in enumerate(json_data):
    text = entry["text"]  # Assuming "text" key has relevant data

    # Convert to Embeddings
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.encoder(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    # Store in ChromaDB
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings],
        metadatas=[entry]  # Store full JSON entry as metadata
    )

print("JSON data stored in ChromaDB successfully ✅")
