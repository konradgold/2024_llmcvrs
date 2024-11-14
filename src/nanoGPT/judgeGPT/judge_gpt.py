import datetime
import sys
from transformers import pipeline
import json
import yaml

class JudgeGPT:
    pipe: pipeline
    judge_model: str = 'meta-llama/Llama-3.2-1B'
    write_to_file: str
    probe_prompts: list
    judge_prompt: str
    deactivate_blocks: list = []

    def __init__(self, model, config_file: str|None, probe_prompts: list, judge_prompt: str, write_dir: str = "judgements/out"):
        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                for key, value in config.items():
                    setattr(self, key, value)
        self.model = model
        self.write_to_file = write_dir + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + ".json"
        self.probe_prompts = probe_prompts
        self.judge_prompt = judge_prompt
        self.pipe = pipeline("text-generation", model=self.judge_model, max_length=10_000)

        # Deactivate blocks if specified
        if len(self.deactivate_blocks)>0:
            self.model.model.deactivate_blocks(self.deactivate_blocks)
        
    def update_blocked(self, blocked: list):
        unblock = [b for b in self.deactivate_blocks if b not in blocked]
        self.model.model.disable_heads(blocked)
        self.model.model.enable_heads(unblock)

    def judge(self):
        judgements = {
            "judgements": [],
            "queries": self.probe_prompts,
            "outputs": self.model.generate_output(text=self.probe_prompts)
        }
        for output in judgements['outputs']:
            PROMPT = self.judge_prompt.format(query=output)
            judgement = self.pipe(PROMPT)
            judgements['judgements'].append(judgement)

        with open(self.write_to_file, 'w') as file:
            json.dump(judgements, file)
        return judgements