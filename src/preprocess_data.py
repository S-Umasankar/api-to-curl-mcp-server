import json
import torch
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

dataset = json.load(open("api_to_curl_dataset.json", "r"))

processed_data = [
    {
        "input_ids": tokenizer(entry["api_documentation"], return_tensors="pt")["input_ids"].squeeze(0),
        "labels": tokenizer(entry["curl_command"], return_tensors="pt")["input_ids"].squeeze(0),
    }
    for entry in dataset
]

torch.save(processed_data, "preprocessed_api_to_curl.pt")
print("Preprocessing Complete ✅")
