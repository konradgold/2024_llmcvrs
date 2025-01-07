import numpy as np
import matplotlib.pylab as plt
from typing import List
from dotenv import load_dotenv
import torch
from nanoGPT import SampleMutableModel
import openai
import random
import wandb
from eval import eval
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

judge = JudgeInstructor(judge_prompt=judge_prompt)

model = SampleMutableModel()

client = openai.OpenAI()

def generate_prompt() -> str:
    response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[
        { "role": "system", "content": "You are a helpful assistant." },
        {
            "role": "user",
            "content": f"Write very short prompt to test a language model's ability to generate text. Let it write about everyday stuff, like school, sports, friends, taxes, work, ... By the way, I am calling you a lot with the same prompt, and you are pretty repetitive (not your fault). So here is a random number, perhabs that will give you some ideas: {random.randint(0, 1000)}",
        },
    ],
    )
    message = response.choices[0].message.content
    assert isinstance(message, str)
    return message


def reduce_model(model, data, knowledge_prompt, number_of_blocks=10, repetitions=5, kill_simultaneously=5):
    reduce_list = []
    try:
        for _ in range(number_of_blocks//kill_simultaneously):
            output: List[str] = []
            prompts: List[str] = []
            activation_norms = {}
            for _ in range(repetitions):
                prompt: str = generate_prompt()
                prompts.append(prompt)
                output += model.generate_output(prompt)
                for block_nr, activations in model.model.attentions.items():
                    for activation in activations:
                        s = activation.shape[2]
                        a = activation.reshape(1, 12, s*64)
                        for j in range(12):
                            if (block_nr, j) in activation_norms:
                                activation_norms[(block_nr,j)] += float(a[0, j, :].view(s*64).norm())
                            else:
                                activation_norms[(block_nr,j)] = float(a[0, j, :].view(s*64).norm())
            
            eval(model, data, knowledge_prompt)
            judgement = judge.judge_output(output, prompts)
            wandb.log({
                "judgement": {
                    "vocabulary": judgement.vocabulary,
                    "grammar": judgement.grammar,
                    "mechanics": judgement.mechanics,
                },
            })
            print(judgement)
            
            #if judgement.mechanics + judgement.grammar + judgement.vocabulary < 6:
            #    print("Judgement was too low, stopping reduction")
            #    yield activation_norms
            #    break
            
            norm_list = list(activation_norms.items())
            norm_list.sort(key=lambda x: x[1])
            kill_nr = 0
            for name, _ in norm_list:
                if (name[0], name[1]) in reduce_list:
                    continue
                else:
                    reduce_list.append((name[0], name[1]))
                    kill_nr += 1
                    if kill_nr >= kill_simultaneously:
                        break
            print(len(reduce_list))
            model.update_blocked(reduce_list)
            wandb.log({
                "blocked_heads": len(reduce_list),
            })
            yield activation_norms
    except GeneratorExit:
        torch.save(model.model.state_dict(), 'model_fresh.pth')

with open('head-to-tail-main/head_to_tail_goodreads.json', 'r') as f:
    data = json.load(f)

prompt = "This is a question to test your factual knowledge. Answer it as concise as possible. Question: "

k = 0
for norms in reduce_model(model, data = data, knowledge_prompt=prompt, number_of_blocks=30, repetitions=4, kill_simultaneously=5):
    data_arr = np.zeros((12,12))
    for (i,j), norm in norms.items():
        data_arr[i,j] = norm
    masked_data = np.ma.masked_where(data_arr == 0., data_arr)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='red')

# Plot heatmap with masking
    plt.imshow(masked_data, cmap=cmap, interpolation='none')
    plt.colorbar(label='Value')
    plt.savefig(f'/Users/konradgoldenbaum/Developement/LLMCVRS/material/heatmap_{k}.png')
    plt.close()
    k+=1



