import vector_db
import chromadb
from transformers import T5ForConditionalGeneration
import torch
from transformers import T5Tokenizer

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")

# Get the chromadb collection
collection = client.get_or_create_collection(name="api_doc")

# Retrieve Data from ChromaDB
results = collection.get(include=["embeddings", "metadatas"])

# Load model and tokenizer
model_path = "models/t5_api_to_curl_v1"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Extract Text and Process with T5
processed_texts = []
for entry in results["metadatas"]:
    text = entry["text"]  # Get stored text
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.encoder(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    # Store processed text
    processed_texts.append({
        "id": entry["id"],
        "processed_embedding": output
    })

print("Pre-processing completed ✅")

# Store Processed Data
for data in processed_texts:
    collection.update(
        ids=[data["id"]],
        embeddings=[data["processed_embedding"]]
    )

print("Processed data updated in ChromaDB!")
