import json
from typing import List
from dotenv import load_dotenv
import torch
from nanoGPT import SampleMutableModel
import openai
from nanoGPT.judgeGPT.judge_instructor import JudgeInstructor

load_dotenv()

with open('./nanoGPT/judgeGPT/prompts/judge_prompts.json', 'r') as file:
    judge_prompts = json.load(file)

judge = JudgeInstructor(judge_prompt=judge_prompts['prompt1'])

model = SampleMutableModel()
model.get_activations()

client = openai.OpenAI()

def generate_prompt():
    response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[
        { "role": "system", "content": "You are a helpful assistant." },
        {
            "role": "user",
            "content": "Write a creative, very short prompt to test a language model's ability to generate text.",
        },
    ],
    )
    return response.choices[0].message.content


def reduce_model(model, number_of_blocks=10, repetitions=5):
    activation_norms = {}
    reduce_list = []
    for _ in range(number_of_blocks):
        min_norm = ("", float("inf"))
        output: List[str] = []
        for _ in range(repetitions):
            prompt = generate_prompt()
            output += model.generate_output(prompt)
            for name, activation in model.activations.items():
                s = activation.shape[1]
                a = activation.view(1, s, 12, 64)
                a = a.permute(0, 2, 1, 3).contiguous()
                for j in range(12):
                    if name in activation_norms:
                        activation_norms[(name,j)] += float(a[:, j, :, :].view(1, s*64).norm(dim=-1))
                    else:
                        activation_norms[(name,j)] = float(a[:, j, :, :].view(1, s*64).norm(dim=-1))
        judgement = judge.judge_output(output)
        print(judgement)
        print(output)
        
        if judgement.mechanics + judgement.content + judgement.grammar + judgement.vocabulary < 8:
            print("Judgement was too low, stopping reduction")
            print(output)
            break

        for name, norm_sum in activation_norms.items():
            if (name[0].split('.')[3], name[1]) in reduce_list:
                continue
            if min_norm[1] > norm_sum:
                min_norm = (name, norm_sum)
        reduce_list.append((int(min_norm[0][0].split('.')[3]), int(min_norm[0][1])))
        activation_norms.clear()
        print(len(reduce_list))

        model.update_blocked(reduce_list)
    torch.save(model.model.state_dict(), 'model.pth')

reduce_model(model, 144, 2)

# Heat map of attentions
# Friday: Non-sensical text



