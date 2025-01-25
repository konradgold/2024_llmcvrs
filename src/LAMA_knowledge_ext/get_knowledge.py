import json
from LAMA_knowledge_ext.knowledge_calculator import SimilarityCalculator
from nanoGPT.sample_model import SampleMutableModel
import random
import tqdm
import torch
import argparse

parser = argparse.ArgumentParser(description="Extract knowledge using a fine-tuned GPT model.")
parser.add_argument("--nr_queries", type=int, default=65, help="Number of queries to sample from each dataset")
parser.add_argument("--model_path", type=str, default='models/finetuned_gpt_040.pt', help="Path to the fine-tuned GPT model")
parser.add_argument("--output_knowledge", type=str, default='LAMA_knowledge_ext/results/knowledge_040.json', help="Path to save the extracted knowledge")
parser.add_argument("--output_similarity", type=str, default='LAMA_knowledge_ext/results/similarity_040.json', help="Path to save the similarity results")
parser.add_argument("--use_llm", action="store_true" ,help="Whether to use the LLM for similarity calculation")

args = parser.parse_args()
nr_queries = args.nr_queries
model_path = args.model_path
output_knowledge = args.output_knowledge
output_similarity = args.output_similarity
use_llm = args.use_llm

print(f"Use llm: {use_llm}")

queries = []


with open("LAMA_knowledge_ext/data/ConceptNet/test.json", "r") as file:
    statements = json.load(file)

querie_new = [(q["masked_sentences"][0].split("[MASK]")[0], q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))

with open("LAMA_knowledge_ext/data/Google_RE/date_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(f'{q["sub_label"]} was born in', q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))

with open("LAMA_knowledge_ext/data/Google_RE/place_of_birth_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(f'{q["sub_label"]} was born in', q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))

with open("LAMA_knowledge_ext/data/Google_RE/place_of_death_test.json", "r") as file:
    statements = json.load(file)

querie_new = [(f'{q["sub_label"]} died in', q["obj_label"]) for q in statements]
queries += random.sample(querie_new, min(nr_queries, len(querie_new)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sm_model = SampleMutableModel()
sm_model.model = torch.load(model_path, weights_only=False, map_location=device)
sm_model.top_k = 10
sm_model.max_new_tokens = 5
knowledge = []
similarity_calc = SimilarityCalculator()
found, not_found = 0, 0
sim_results = []
for query, truth in tqdm.tqdm(queries):
    try:
        out, probs, tokens = sm_model.generate_top_k_samples(query, 5)
        predictions = []
        for i, o in out.items():
            response: str = sm_model.decode(o.tolist()[0]).replace(query, "")
            predictions.append(response.replace(query, ""))
        sim = similarity_calc.calculate_similarity(query, probs, predictions, truth, use_llm=use_llm)
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
    
with open(output_knowledge, "w") as file:
    json.dump(knowledge, file, indent=4)

with open(output_similarity, "w") as file:
    json.dump(sim_results, file, indent=4)

print(f"Found: {found}, Not Found: {not_found}")