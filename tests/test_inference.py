import requests

def test_inference():
    url = "http://127.0.0.1:8000/generate_curl/"
    data = {"api_text": "GET /users/{id}"}
    response = requests.post(url, json=data)
    assert response.status_code == 200, "Inference API failed"
    assert "curl_command" in response.json(), "Missing key: curl_command"

test_inference()
print("✅ Inference Test Passed!")
