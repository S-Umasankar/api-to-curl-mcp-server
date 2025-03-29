from transformers import T5ForConditionalGeneration
import torch
from transformers import T5Tokenizer
import chromadb

# Load model and tokenizer
model_path = "models/t5_api_to_curl_v1"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Define Query for RAG
query = "Explain API authentication methods"
query_inputs = tokenizer(query, return_tensors="pt")

# Convert Query into Embeddings
with torch.no_grad():
    query_embedding = model.encoder(**query_inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")

# Get a collection in ChromaDB
collection = client.get_or_create_collection(name="api_doc")

# Retrieve Top 3 Relevant Documents
results = collection.query(query_embeddings=[query_embedding], n_results=3)

# Extract Relevant Texts
retrieved_texts = [entry["text"] for entry in results["metadatas"]]

# Generate Response Using T5 Model
context = " ".join(retrieved_texts)
input_text = f"question: {query} context: {context}"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate Answer
output_ids = model.generate(input_ids)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Response:", answer)
