import requests

def test_inference():

    print("Inference Testing starts....")
    url = "http://127.0.0.1:8000/docs/generate_curl/"
    data = {"api_text": "GET /users/{id} Retrieves user details"}
    response = requests.post(url, json=data)

    if response.status_code == 200 and "curl_command" in response.json():
        print("✅ Inference Test Passed! Generated cURL:", response.json()["curl_command"])
    else:
        print("❌ Inference Test Failed!", response.text)

if __name__ == "__main__":
    test_inference()





