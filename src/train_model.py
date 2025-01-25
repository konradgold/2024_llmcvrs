from finetune import finetune, CustomDataset
import json
import torch
from openai import OpenAI
from datasets import load_dataset
from infra_reduce_model import evaluate_model, TextDataLoader

with open ("filter-openwebtext/knowledge_texts.json", "r") as file:
    data = json.load(file)

dataset = CustomDataset(data)

model = torch.load("models/finetuned_gpt_0.05.pt", weights_only=False)

for param in model.parameters():
    torch.nn.init.normal_(param)

model_new = finetune(model, None, dataset, epochs=30)

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
dataloader_validation = TextDataLoader(load_dataset("mintujupally/ROCStories", split="test")["text"])

model.eval()
        # 2) Evaluate the model
y_value = evaluate_model(model, client, dataloader_validation)
print(f"Final Score: {y_value}")

torch.save(model_new, "models/gpt_0.05_trained.pt")

