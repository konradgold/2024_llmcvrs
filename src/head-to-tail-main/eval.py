import json
from rouge_score import rouge_scorer
import wandb

EVAL_DS = "DBLP"

wandb.init(
    project="llmcvrs2024",
    config={
        "dataset": EVAL_DS,
        "model": "GPT-2-small",
    }
)

def eval(model, data):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # Example function to evaluate DBLP data
    scores = []
    for entry in data["head"]:
        response = model.generate_output(entry.question)
        score = scorer.score(response, entry.ground_truth)
        scores.append(score["rougeL"].fmeasure)

    wandb.log({
        "mean rougeL-fmeasure": sum(scores) / len(scores)
    })




