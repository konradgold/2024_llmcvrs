import json
import sys
import torch
sys.path.append("/Users/konradgoldenbaum/Developement/LLMCVRS/src/nanoGPT")
from sample_model import SampleMutableModel
import random
import tqdm
import Levenshtein

queries = []

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/ConceptNet/test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(200, len(querie_new)))
print(len(querie_new))

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Squad/test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(200, len(querie_new)))
print(len(querie_new))

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Google_RE/date_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(200, len(querie_new)))
print(len(querie_new))

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Google_RE/place_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(200, len(querie_new)))
print(len(querie_new))


with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Google_RE/place_of_death_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(200, len(querie_new)))

model = SampleMutableModel()
#state_dict = torch.load('/Users/konradgoldenbaum/Developement/LLMCVRS/src/model.pth', weights_only=False)
#model.model.load_state_dict(state_dict)
knowledge = []

found, not_found = 0, 0
for query, truth in tqdm.tqdm(queries):
    out, probs = model.generate_top_k("The capital of France is ", 10)
    predictions = [model.decode([o]) for o in out[0].tolist()]
    if min(Levenshtein.distance(p, truth) for p in predictions) < 3:
        knowledge.append({
                    'sentence': query,
                    'object_ground_truth_idx': truth,
                    'object_predicted': predictions[0],
                    'object_predicted_10': predictions,
                    'filtered_log_probs': probs[0].tolist(),
                })
        found += 1
    else:
        not_found += 1
    
with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/knowledge.json", "w") as file:
    json.dump(knowledge, file, indent=4)

print(f"Found: {found}, Not Found: {not_found}")



