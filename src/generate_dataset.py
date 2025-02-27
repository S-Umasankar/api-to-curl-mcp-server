import json
import random


def generate_api_sample(sample_id):
    http_methods = ["GET", "POST", "PUT", "DELETE"]
    resources = ["users", "orders", "products", "payments"]
    method = random.choice(http_methods)
    resource = random.choice(resources)
    endpoint = f"/{resource}/{random.randint(1000, 9999)}" if method != "POST" else f"/{resource}"

    api_doc = f"{method} {endpoint} - Manage {resource}."
    curl_command = f'curl -X {method} "https://api.example.com{endpoint}" -H "Authorization: Bearer TOKEN"'

    return {"id": sample_id, "api_documentation": api_doc, "curl_command": curl_command}


dataset = [generate_api_sample(i) for i in range(1, 501)]
json.dump(dataset, open("data/input/api_to_curl_dataset.json", "w"), indent=4)
print("Dataset Generated ✅")
