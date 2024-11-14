import datetime
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import yaml
from model import GPTConfig, GPT
import json


class SampleModel:
    init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    write_file = "out/write_"
    out_dir = 'out' # ignored if init_from is not 'resume'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 3 # number of samples to draw
    max_new_tokens = 50 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster

    def __init__(self, init_from: str = 'gpt2', config_file: str | None = None):
        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                for key, value in config.items():
                    setattr(self, key, value)
        
        self.write_file = self.write_file + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + ".json"

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if init_from == 'resume':
            ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            self.model = GPT.from_pretrained(init_from, dict(dropout=0.0))

        self.model.eval()
        self.model.to(self.device)
        if compile:
            self.model = torch.compile(self.model)
        self.load_meta = False
        if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
            self.meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
            self.load_meta = os.path.exists(self.meta_path)
        if self.load_meta:
            print(f"Loading meta from {self.meta_path}...")
            with open(self.meta_path, 'rb') as f:
                self.meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = self.meta['stoi'], self.meta['itos']
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)

    
    
    def update_blocked(self, blocked: list):
        if self.deactivate_blocks:
            unblock = [b for b in self.deactivate_blocks if b not in blocked]
            self.model.model.enable_heads(unblock)
        self.deactivate_blocks = blocked
        self.model.model.disable_heads(self.deactivate_blocks)

        


    def generate(self, text: str):
        text = self._get_text(text)
        output = []
        for s in text:
            start_ids = self.encode(s)
            x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
            with torch.no_grad():
                with self.ctx:
                    y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                    output.append({s: self.decode(y[0].tolist())})
                
        # Load existing content if the file already exists
        self.write_output(output)
    
    def generate_output(self, text: str):
        text = self._get_text(text)
        output = []
        for s in text:
            start_ids = self.encode(s)
            x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
            with torch.no_grad():
                with self.ctx:
                    y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                    output.append({s: self.decode(y[0].tolist())})
        
        return output
        
    
    def generate_verbose(self, text: str):
        text = self._get_text(text)

        for s in text:
            start_ids = self.encode(s)
            x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
            with torch.no_grad():
                with self.ctx:
                    y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                    print({s: self.decode(y[0].tolist())})

        
    def _get_text(self, text: str) -> list:
        if isinstance(text, str) and text.startswith('FILE:'):
            with open(text[5:], 'r', encoding='utf-8') as f:
                text = f.read()
    
        if not isinstance(text, list):
            text = [text]
        
        return text

    def write_output(self, output: list):
        if os.path.exists(self.write_file):
            with open(self.write_file, 'r', encoding='utf-8') as f:
                try:
                    existing_content = json.load(f)
                except json.JSONDecodeError:
                    existing_content = []
        else:
            existing_content = []

        # Append new output to the existing content
        existing_content.extend(output)

        # Write the updated content back to the file
        with open(self.write_file, 'w', encoding='utf-8') as f:
            json.dump(existing_content, f, ensure_ascii=False, indent=4)
