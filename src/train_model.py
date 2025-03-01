"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from contextlib import nullcontext
from nanoGPT.model import GPTConfig, GPT
import numpy as np
from nanoGPT.prune_model import prune_attention, prune_model
import torch
import json
import argparse

parser = argparse.ArgumentParser(description="Extract knowledge using a fine-tuned GPT model.")
parser.add_argument("--init_from", type=str, default="custom", help="Checkpoint or first training")
parser.add_argument("--train_dataset", type=str, default='filter-openwebtext/filter_folder/train_0.2.bin', help="Path to the fine-tuned GPT model")
parser.add_argument("--validation_dataset", type=str, default='nanoGPT/data/openwebtext/val.bin', help="Path to save the extracted knowledge")
parser.add_argument("--output_dir", type=str, default='out/out02', help="Path to save the similarity results")
parser.add_argument("--prune_percent", type=str, default="reduction_scores/results020_finetune_mlp_weight.json", help="Pruning degree")
parser.add_argument("--prune_part", type=str, default="mlp", help="mpl or attention")

args = parser.parse_args()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

out_dir = args.output_dir
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = args.init_from # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'llmcvrs2024'
wandb_run_name = 'run' + str(time.time())
model_path=None
chkpt_file=None
# data
train_data_dir = args.train_dataset
val_data_dir = args.validation_dataset
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = "cuda" if torch.cuda.is_available() else "cpu" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(train_data_dir, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(val_data_dir, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'resume' and args.prune_percent is not None:
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        model = checkpoint['model']
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        iter_num = 0
        model = checkpoint
elif init_from == "custom" and args.prune_percent is not None:
    model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
    for param in model.parameters():
        torch.nn.init.normal_(param)
    with open(args.prune_percent, 'r') as f:
        prune_percent = json.load(f)
    prune_percent = prune_percent["X"][np.argmax(prune_percent["Y"])]
    if args.prune_part == "mlp":
        model = prune_model(model, prune_percent)
    else:
        model = prune_model(model, prune_percent, attention=True)
    iter_num = 0
    
    
# crop down the model block size if desired, using model surgery
model.to(device)
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda",enabled=(dtype == 'float16'))

from google.genai import Client
from datasets import load_dataset
from nanoGPT.sample_model import SampleMutableModel
from dotenv import load_dotenv, find_dotenv
import random

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

class TextDataLoader:
    """
    A simple dataloader for text data that allows sampling random batches of texts.
    """
    def __init__(self, texts: list[str]):
        self.texts = texts

    def get_random_batch(self, batch_size):
        """
        Returns a random batch of texts.

        Args:
            batch_size (int): Number of texts to return in the batch.

        Returns:
            list: A random batch of texts.
        """
        if batch_size > len(self.texts):
            raise ValueError("Batch size exceeds the number of texts available.")
        return random.sample(self.texts, batch_size)

def check_model_language_performance(model, client, dataset, runs=10):
    batch = dataset.get_random_batch(runs)
    output = model.generate_output(batch)
    judgements = []
    for t in output:
        judgements.append(client.models.generate_content(
        model="gemini-1.5-flash-latest",
        contents=judge_prompt.format(response=t)
        ).text)
    
    return judgements

load_dotenv(find_dotenv())

# Access environment variables
api_key = os.getenv('GEMINI_API_KEY')
client = Client(api_key=api_key)

ds = load_dataset("mintujupally/ROCStories", split="test")
dataloader_validation = TextDataLoader(texts=ds["text"])

sm_model = SampleMutableModel(model=model)
judgement_scores = check_model_language_performance(sm_model, client, dataloader_validation)
print(f"Judgement scores: {judgement_scores}")
with open('judgement_scores.txt', 'a') as f:
    for score in judgement_scores:
        f.write(score + '\n')

print("Exiting test")
exit()

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    print(wandb.__file__)
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

time_start = time.time()
time_start /= 3600

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    time_end = time.time()
    time_end /= 3600
    if time_end - time_start >= 22.5:
        try:
            sm_model = SampleMutableModel(model=model)
            judgement_scores = check_model_language_performance(sm_model, client, dataloader_validation)
            print(f"Judgement scores: {judgement_scores}")
            with open('judgement_scores.txt', 'a') as f:
                for score in judgement_scores:
                    f.write(score + '\n')
            print(f"Shutting down after {time_end - time_start} hours")
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model,
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {args.output_dir}")
                torch.save(checkpoint, os.path.join(args.output_dir, 'ckpt.pt'))
        except:
            print("Error during evaluation")

    if iter_num % eval_interval == 0 and master_process:
        try:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {args.output_dir}")
                    torch.save(checkpoint, os.path.join(args.output_dir, 'ckpt.pt'))
        except Exception as e:
            print(f"error during evaluation: {e}")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

