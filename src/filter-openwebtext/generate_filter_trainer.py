import json
import argparse

parser = argparse.ArgumentParser(description="Extract dataset using a fine-tuned GPTs model.")
parser.add_argument("--similarity", type=str, default="LAMA_knowledge_ext/results/similarity_040.json", help="Number of queries to sample from each dataset")
parser.add_argument("--texts", type=str, default="filter-openwebtext/knowledge_texts.json", help="Path to the fine-tuned GPT model")


args = parser.parse_args()
file_path = args.similarity
out_path = args.texts
# Function to load JSON data from a file

with open(file_path, 'r') as file:
    data = json.load(file)

train_texts = []

for item in data:
    rogue_mean = sum([(sum(rogue["rouge1"]) + sum(rogue["rougeL"]))/6 for rogue in item["rogue"]])/len(item["rogue"])
    bleu_mean = sum(item["bleu"])/len(item["bleu"])
    loss = item["loss"]
    llm_judgement = item["llm_judgement"]
    found = max([item["truth"] in prediction for prediction in item["predictions"]])
    if found or bleu_mean > 1e-10 or rogue_mean > 1e-2 or llm_judgement > 1e-2 or loss < 0.7:
        train_texts.append(item["query"].strip() + " " + item["truth"])

print(f"Found {len(train_texts)} texts")

with open(out_path, "w") as file:
    json.dump(train_texts, file)
