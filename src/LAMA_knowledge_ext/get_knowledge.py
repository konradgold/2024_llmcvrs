import json
from LAMA_knowledge_ext.knowledge_calculator import SimilarityCalculator
from nanoGPT.sample_model import SampleMutableModel
import random
import tqdm
import torch

queries = []

nr_queries = 50

with open("LAMA_knowledge_ext/data/ConceptNet/test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(querie_new[0])
print(len(querie_new))

with open("LAMA_knowledge_ext/data/Google_RE/date_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(f"{q["sub_label"]} was born in", q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(querie_new[0])
print(len(querie_new))

with open("LAMA_knowledge_ext/data/Google_RE/place_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(f"{q["sub_label"]} was born in", q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(querie_new[0])
print(len(querie_new))

with open("LAMA_knowledge_ext/data/Google_RE/place_of_death_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(f"{q["sub_label"]} died in", q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))

print(querie_new[0])
print(len(queries))

model = SampleMutableModel()
model.top_k = 10
model.max_new_tokens = 5
state_dict = torch.load('finetuned_gpt_IT.pt', weights_only=False)
model.model.load_state_dict(state_dict, strict=True)
knowledge = []
similarity_calc = SimilarityCalculator()
found, not_found = 0, 0
sim_results = []
for query, truth in tqdm.tqdm(queries):
    try:
        out, probs, tokens = model.generate_top_k_samples(query, 5)
        predictions = []
        for i, o in out.items():
            predictions.append(model.decode(o.tolist()[0]).replace(query, ""))
        sim = similarity_calc.calculate_similarity(query, probs, predictions, truth)
        sim["query"] = query
        sim["truth"] = truth
        sim["predictions"] = predictions
        sim_results.append(sim)
        for pred in predictions:
            if truth in pred:
                knowledge.append({
                        'sentence': query,
                        'object_ground': truth,
                        'object_predicted': pred,
                        'object_predicted_10': predictions,
                    })
                found += 1
                break
        not_found += 1
    except Exception as e:
        print(e)
        print(f"Failed for {query + truth}")
    
with open("LAMA_knowledge_ext/knowledge_IT.json", "w") as file:
    json.dump(knowledge, file, indent=4)

with open("LAMA_knowledge_ext/similarity_IT.json", "w") as file:
    json.dump(sim_results, file, indent=4)

print(f"Found: {found}, Not Found: {not_found}")



