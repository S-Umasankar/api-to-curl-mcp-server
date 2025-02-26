import time
import requests
import subprocess
from git import Repo

# MCP Server URL
MCP_SERVER_URL = "http://127.0.0.1:8000"

# GitHub Repo Info
GITHUB_REPO_PATH = "/Users/umasankars/PycharmProjects/CapstoneMCPserver/"
GITHUB_REMOTE_URL = "https://github.com/S-Umasankar/api-to-curl-mcp-server.git"


def call_api(endpoint):
    """Helper function to call MCP server API endpoints."""
    try:
        response = requests.get(f"{MCP_SERVER_URL}{endpoint}")
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def generate_and_test_code():
    """Generate improved code, test it, and deploy it."""
    print("\n🧠 AI Generating New Code for Model Improvements...\n")

    # 1️⃣ Generate new model training code (via OpenAI API or internal logic)
    new_code = """import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")
optimizer = AdamW(model.parameters(), lr=3e-5)
# Model training code improved with dynamic learning rate...
"""
    with open("src/train_model.py", "w") as f:
        f.write(new_code)

    print("✅ New code generated and saved.")

    # 2️⃣ Run the generated code as a test
    print("\n⚙️ Running Tests on New Code...\n")
    try:
        subprocess.run(["python", "train_model.py"], check=True)
        print("✅ Code executed successfully. Ready for deployment.")
    except subprocess.CalledProcessError:
        print("❌ Test failed. AI will debug and retry.")
        return False

    return True


def deploy_and_push_code():
    """Deploy the updated code and push changes to GitHub."""
    print("\n🚀 Deploying the updated code...\n")

    # 1️⃣ Run the MCP Server with the new model
    subprocess.run(["uvicorn", "mcp_server:app", "--reload"])

    # 2️⃣ Push updated code to GitHub
    print("\n📤 Pushing new changes to GitHub...\n")
    repo = Repo(GITHUB_REPO_PATH)
    repo.git.add(A=True)
    repo.index.commit("AI Auto-Update: Model & Training Code Improved")
    repo.remote(name="origin").push()

    print("✅ Code pushed to GitHub successfully.")


def auto_train_loop():
    """Continuous AI-driven automation loop."""
    while True:
        print("\n🔄 Starting AI-Driven Execution Cycle...\n")

        # 1️⃣ Generate new dataset
        print("🟢 Generating dataset...")
        print(call_api("/generate_dataset/"))
        time.sleep(5)

        # 2️⃣ Preprocess Data
        print("🟢 Preprocessing dataset...")
        print(call_api("/preprocess_data/"))
        time.sleep(5)

        # 3️⃣ Train the Model
        print("🟢 Training model...")
        print(call_api("/train_model/"))
        time.sleep(60 * 30)  # Wait 30 minutes

        # 4️⃣ Check Performance (BLEU Score)
        print("🟢 Evaluating model performance...")
        bleu_score = requests.get(f"{MCP_SERVER_URL}/evaluate_model/").json()
        print(f"🟢 BLEU Score: {bleu_score}")

        # 5️⃣ Fine-Tune If Needed
        if bleu_score['score'] < 50:
            print("🔴 Low BLEU Score detected! Auto fine-tuning the model...")
            print(call_api("/auto_finetune/"))
            time.sleep(60 * 15)  # Wait for fine-tuning (~15 min)

        # 6️⃣ AI-Generated Code Updates
        if generate_and_test_code():
            deploy_and_push_code()

        print("\n✅ AI Execution Cycle Complete! Waiting for the next cycle...\n")
        time.sleep(60 * 60 * 6)  # Wait 6 hours before next cycle


# Start AI-driven automation
auto_train_loop()