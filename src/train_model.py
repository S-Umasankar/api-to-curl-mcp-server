import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

dataset = torch.load("data/preprocessed_api_to_curl.pt")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids, labels = batch["input_ids"], batch["labels"]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")

model.save_pretrained("models/t5_api_to_curl")
tokenizer.save_pretrained("models/t5_api_to_curl")
print("Training Complete ✅")
