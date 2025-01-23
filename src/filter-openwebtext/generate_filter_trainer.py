import json

file_path = "LAMA_knowledge_ext/similarity_IT.json"
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
    if found or bleu_mean > 1e-10 or rogue_mean > 1e-2 or llm_judgement > 1e-2 or loss < 0.9:
        train_texts.append(item["query"].strip() + " " + item["truth"])

print(f"Found {len(train_texts)} texts")

with open("filter-openwebtext/knowledge_texts.json", "w") as file:
    json.dump(train_texts, file)
