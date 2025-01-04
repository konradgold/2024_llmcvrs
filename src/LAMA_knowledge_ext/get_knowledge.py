import json
import sys
sys.path.append("/Users/konradgoldenbaum/Developement/LLMCVRS/src/nanoGPT")
from sample_model import SampleMutableModel
import random
import tqdm

queries = []

nr_queries = 50

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/ConceptNet/test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(len(querie_new))

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Squad/test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(len(querie_new))

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Google_RE/date_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(len(querie_new))

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Google_RE/place_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))
print(len(querie_new))


with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/data/Google_RE/place_of_death_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))

model = SampleMutableModel()
#state_dict = torch.load('/Users/konradgoldenbaum/Developement/LLMCVRS/src/model.pth', weights_only=False)
#model.model.load_state_dict(state_dict)
knowledge = []

found, not_found = 0, 0
for query, truth in tqdm.tqdm(queries):
    try:
        out = model.generate_top_k(query, 5)
        predictions = []
        for i, o in out.items():
            predictions.append(model.decode(o.tolist()[0]).replace(query, ""))
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
    except:
        print(f"Failed for {query}")
        continue
    
with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/knowledge.json", "w") as file:
    json.dump(knowledge, file, indent=4)

print(f"Found: {found}, Not Found: {not_found}")



