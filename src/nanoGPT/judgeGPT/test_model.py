import datetime
import sys
from transformers import pipeline
import json
import argparse

sys.path.append('./..')
from sample_model import SampleModel

parser = argparse.ArgumentParser(description='Process some prompts.')
parser.add_argument('--prompt', type=str, required=False, help='Path to the prompts JSON file', default='prompt1')
parser.add_argument('--model', type=str, required=False, help='Judge model to use (Huggingface)', default='meta-llama/Llama-3.2-1B')
parser.add_argument('--write_dir', type=str, required=False, help='Path to the output JSON file', default='judgements/out' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") +'.json')

model = SampleModel(config_file='../config/eval_gpt2.yaml')

args = parser.parse_args()

with open('prompts/judge_prompts.json', 'r') as file:
    prompt = json.load(file)[args.prompt]

with open('prompts/file1.json', 'r') as file:
    queries = json.load(file)

outputs = model.generate_output(text=queries)

judgements = {
    "judgements": [],
    "queries": queries,
    "outputs": outputs
}

pipe = pipeline("text-generation", model=args.model, max_length=10_000)
for output in outputs:
    PROMPT = prompt.format(query=output)
    print(PROMPT)
    judgement = pipe(PROMPT)
    print(judgement)
    judgements['judgements'].append(judgement)

with open(args.write_dir, 'w') as file:
    json.dump(judgements, file)