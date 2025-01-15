import json
from rouge_score import rouge_scorer
import wandb
from tqdm import tqdm
import random

EVAL_DS = "goodreads"

wandb.init(
    project="llmcvrs2024",
    config={
        "dataset": EVAL_DS,
        "model": "GPT-2-small",
    }
)

def eval(model, data, prompt, select=0.1):
    model.model.max_new_tokens = 15
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # Example function to evaluate DBLP data
    scores = []
    selection = random.sample(data["head"], int(len(data["head"]) * select))
    for entry in tqdm(selection):
        response = ""
        try:
            response = model.generate_output(prompt + entry[2] + "Answer: ")[0]
            response = response.replace(prompt, '')
            response = response.replace('Answer: ', '')
            response = response.replace(entry[2], '')
            response = response.replace("\n", "").strip()
            if isinstance(entry[3], list):
                entry[3] = ", ".join(entry[3])
            score = scorer.score(response, entry[3])
            scores.append(score["rougeL"].fmeasure)

            wandb.log({
                "rougeL-fmeasure": score["rougeL"].fmeasure
            })
        except Exception as e:
            print(e)
            print(response)
            print(entry)
            continue
    wandb.summary["Mean rougeL-fmeasure"] = sum(scores) / len(scores)
    model.model.max_new_tokens = 100


def eval_similarity(true_story, generated_story):
    scorer = rouge_scorer.RougeScorer(['rougeL', "rouge1"], use_stemmer=True)
    return scorer.score(true_story, generated_story)




