import torch
import sacrebleu
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model_path = "models/t5_api_to_curl"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load dataset
dataset = torch.load("data/input/preprocessed_api_to_curl.pt")

# Evaluate model
references, predictions = [], []
for sample in dataset[:10]:  # Use first 10 samples for testing
    input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    reference_curl = tokenizer.decode(sample["labels"], skip_special_tokens=True)

    # Generate cURL command
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)
    output_ids = model.generate(input_ids)
    generated_curl = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    references.append([reference_curl])
    predictions.append(generated_curl)

# Calculate BLEU Score
bleu = sacrebleu.corpus_bleu(predictions, references)
print(f"🟢 BLEU Score: {bleu.score}")
