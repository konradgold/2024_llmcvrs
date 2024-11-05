import datetime
from transformers import pipeline
import json
import argparse

parser = argparse.ArgumentParser(description='Process some prompts.')
parser.add_argument('--prompt', type=str, required=False, help='Path to the prompts JSON file', default='prompt1')
parser.add_argument('--model', type=str, required=False, help='Judge model to use (Huggingface)', default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--write_dir', type=str, required=False, help='Path to the output JSON file', default='judgements/out' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") +'.json')

args = parser.parse_args()

with open('prompts/judge_prompts.json', 'r') as file:
    prompt = json.load(file)[args.prompt]


with open('prompts/file1.json', 'r') as file:
    queries = json.load(file)

judgements = {
    "judgements": [],
    "queries": queries
}
PROMPTS = []
for query in queries:
    PROMPTS.append({"role": "user", "content":prompt.format(query=query)})

pipe = pipeline("text-generation", model=args.model, max_length=2000)
judgement = pipe(PROMPTS)
print(judgement)
judgements['judgements'].append(judgement)

with open(args.write_dir, 'w') as file:
    json.dump(judgements, file)