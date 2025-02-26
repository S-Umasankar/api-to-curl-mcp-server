import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def test_training():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Sample input
    input_text = "GET /users/{id}"
    target_text = "curl -X GET 'https://api.example.com/users/123'"

    # Tokenization
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    target_ids = tokenizer.encode(target_text, return_tensors="pt")

    # Forward pass
    outputs = model(input_ids=input_ids, labels=target_ids)
    assert outputs.loss is not None, "Training should produce a loss value"

test_training()
print("✅ Training Test Passed!")
