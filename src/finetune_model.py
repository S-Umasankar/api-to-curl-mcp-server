import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW

# Load model and tokenizer
model_path = "models/t5_api_to_curl"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Load preprocessed dataset
dataset = torch.load("data/preprocessed_api_to_curl.pt")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Fine-tuning loop
for epoch in range(1):
    total_loss = 0
    for batch in train_loader:
        input_ids, labels = batch["input_ids"], batch["labels"]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Fine-Tuning Loss: {total_loss/len(train_loader)}")

# Save fine-tuned model
model.save_pretrained("models/t5_api_to_curl_finetuned")
print("✅ Fine-Tuning Complete")
