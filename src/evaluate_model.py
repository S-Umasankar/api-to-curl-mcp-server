import torch
import sacrebleu
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model_path = "models/t5_api_to_curl_v1"
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

def compute_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {}

    # Make sure references and hypotheses are lists of strings
    for reference, hypothesis in zip(references, hypotheses):
        score = scorer.score(reference[0], hypothesis)
        scores[reference[0]] = score

    return scores

# Compute ROUGE score
rouge_scores = compute_rouge(references, predictions)

# Print results
for metric, score in rouge_scores.items():
    print(f"🟢 Reference: {metric}")
    print(f"🟢 ROUGE Scores: {score}\n")