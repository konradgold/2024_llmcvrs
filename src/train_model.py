from finetune import finetune, CustomDataset
import json
import torch
from google import genai
from datasets.load import load_dataset
from infra_reduce_model import evaluate_model, TextDataLoader
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# Access environment variables
api_key = os.getenv('GEMINI_API_KEY')

with open ("filter-openwebtext/knowledge_texts.json", "r") as file:
    data = json.load(file)

dataset = CustomDataset(data)

model = torch.load("models/finetuned_gpt_0.05.pt", weights_only=False)

for param in model.parameters():
    torch.nn.init.normal_(param)

model_new = finetune(model, None, dataset, epochs=30)

client = genai.Client(api_key)
dataloader_validation = TextDataLoader(load_dataset("mintujupally/ROCStories", split="test")["text"])

model.eval()
        # 2) Evaluate the model
y_value = evaluate_model(model, client, dataloader_validation)
print(f"Final Score: {y_value}")

torch.save(model_new, "models/gpt_0.05_trained.pt")

