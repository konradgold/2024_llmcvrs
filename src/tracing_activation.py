from dotenv import load_dotenv
from nanoGPT import SampleMutableModel
from nanoGPT.prune_model import prune_model
import wandb
from eval import eval_similarity
import json
from nanoGPT.judgeGPT.judge_instructor import JudgeInstructor

EVAL_DS = "goodreads"

wandb.init(
    project="llmcvrs2024",
    config={
        "dataset": EVAL_DS,
        "model": "GPT-2-small",
        "sample_size": 0.1,
    }
)

load_dotenv()

judge_prompt="""Instructions:

Please evaluate the response to the following query according to the specified criteria.

	1.	Assign a score (1-4) for each component based on the criteria provided.

Response to Evaluate:

{response}

Evaluation Criteria:

	1.	Vocabulary (V):
	•	4 (Excellent): Good vocabulary choice.
	•	3 (Good): Few errors in vocabulary choice that do not interfere with understanding. "Snorkeling means not breathing when skiing."
	•	2 (Fair): Errors in vocabulary choice are present interfere with understanding. "I sit on a blue and drink alter."
	•	1 (Poor): Many errors in vocabulary choice that make understanding impossible. "I sit why o read"
	2.	Grammar (G):
	•	4 (Excellent): Good grammar.
	•	3 (Good): Few errors in grammar that do not really interfere with understanding. "I had cook a soup."
	•	2 (Fair): Errors in grammar are present making understanding hard sometimes. "I and we want why not to go to theater"
	•	1 (Poor): The text is essentially unreadable because of the grammar, e.g., "I doesnt apple because implied".
	3.	Mechanics (M):
	•	4 (Excellent): Good spelling, punctuation, and capitalization.
	•	3 (Good): Few errors in spelling, punctuation, and capitalization.
	•	2 (Fair): Errors in spelling, punctuation, and capitalization are present and sometimes interfere with understanding.
	•	1 (Poor): Many errors in spelling, punctuation, and capitalization appear seemingly randomly.

Try not to be too strict. E.g., "Well, you've got to start at the beginning and then you have to get going as quickly as possible." has at least a 3 in all categories.
Also, the fact that the last sentence might end abrupt is not the fault of the model but a technical necessity.

Output Format:

	1.	Vocabulary (V): Score = X
	2.	Grammar (G): Score = X
	3.	Mechanics (M): Score = X
"""
model = SampleMutableModel()
judge = JudgeInstructor(judge_prompt=judge_prompt)

def reduce_model_activation_based(model, prune_percent, runs: int = 1):
    for _ in range(runs):
        prompt = "Finish this story:\n\n"
        storie_start = "Once upon a time"
        model.generate_output(prompt + storie_start)
    model.model = prune_model(model.model, prune_percent, True)
    return model

def check_model_language_performance(model, judge, runs = 10):
    for _ in range(runs):
        prompt = "Finish this story:\n\n"
        storie = "once upon a time, there was a princess who lived in a castle"
        storie_start = " ".join(storie.split(" ")[:len(storie.split(" "))//2])
        print(storie_start)
        output = model.generate_output(prompt + storie_start)
        rouge_score = eval_similarity(storie, output[0])
        print(rouge_score)
        judgement = judge.judge_output(output, [prompt])
        print(judgement.model_dump())

def store_judgement(judgement):
    with open("judgements.json", "r") as f:
        judgements = json.load(f)
    judgements.append(judgement)
    with open("judgements.json", "w") as f:
        json.dump(judgements, f)
    

model.model = prune_model(model.model, 0.1)
check_model_language_performance(model, judge, runs = 1)



