
import sys
from transformers import pipeline
import json
from judge_gpt import JudgeGPT

sys.path.append('./..')
from sample_model import SampleModel

with open('prompts/judge_prompts.json', 'r') as file:
    judge_prompt = json.load(file)[0]

with open('prompts/file1.json', 'r') as file:
    probe_prompts = json.load(file)

model = SampleModel(config_file='../config/eval_gpt2.yaml')

judge = JudgeGPT(model=model, config_file='../config/judge_gpt.yaml', probe_prompts=probe_prompts, judge_prompt=judge_prompt, write_dir='judgements/out')

judgements = judge.judge()
print(judgements)