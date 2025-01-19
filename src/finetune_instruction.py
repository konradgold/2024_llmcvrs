import json
import tiktoken
import torch
from nanoGPT.sample_model import SampleMutableModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5

class CustomDataset_IT(Dataset):
    def __init__(self, data, max_length=300):
        """
        data: list of {'question': str, 'answer': str}
        """
        self.max_length = max_length
        self.enc = tiktoken.get_encoding("gpt2")
        self.examples = []
        
        for ex in data:
            question = ex["question"]
            answer = ex["answer"]
            
            # Build a single text that includes both question & answer
            # so the model sees them in context. For example:
            text = f"Instruction: {question}\nResponse: {answer}"
            
            # Tokenize
            encoded = self.enc.encode(text, allowed_special={"<|endoftext|>"})
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            
            # Make input_ids
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            input_ids[:len(encoded)] = torch.tensor(encoded, dtype=torch.long)
            
            # Make labels
            labels = torch.full((self.max_length,), -100, dtype=torch.long)
            
            # The portion that corresponds to the answer gets actual labels
            # We'll figure out where the answer starts. 
            # For instance, let's encode just the prefix "Instruction: {question}\nResponse:" 
            prefix_text = f"Instruction: {question}\nResponse:"
            prefix_enc = self.enc.encode(prefix_text, allowed_special={"<|endoftext|>"})
            prefix_len = len(prefix_enc)
            
            answer_start = min(prefix_len, self.max_length)
            answer_end = len(encoded)
            
            if answer_start < answer_end:
                labels[answer_start:answer_end] = input_ids[answer_start:answer_end]
            
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # returns a tuple (input_ids, labels)
        return self.examples[idx]

# Path to the JSON file
file_path = 'lernerstories/data/generated_instructions.json'

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

dataset = CustomDataset_IT(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


sample_model = SampleMutableModel()
model = sample_model.model
print(model)

for param in model.transformer.wte.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    # If it's in the last N blocks, allow training; otherwise freeze
    if "transformer.h.10" in name or "transformer.h.11" in name:  # example for last 2 blocks
        print(f"Training: {name}")
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=1e-6)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(dataloader)*epochs)

model_orig = copy.deepcopy(model)
model_orig.to(device)
model_orig.eval()
model.train()
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
kl_weight = 0.01
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        # batch is a tuple of (input_ids, labels) with shape [batch_size, max_length]
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # forward pass
        # The second argument to `model()` is typically "targets" for the built-in loss,
        # but we won't use that. We'll rely on the raw logits from `outputs`.
        outputs, _ = model(input_ids, input_ids)
        
        # ^ We IGNORE the second item (the built-in `loss`), because we want to do our own mask.

        # Suppose `outputs` has shape [batch_size, seq_len, vocab_size]
        logits = outputs  # or outputs.logits if needed
        orig_logits,_ = model_orig(input_ids)

        kl_loss = torch.nn.functional.kl_div(
            input=torch.nn.functional.softmax(logits, dim=-1),
            target=torch.nn.functional.softmax(orig_logits, dim=-1),
            reduction="batchmean"
        )

        # The typical shape for CrossEntropyLoss is [batch_size * seq_len, vocab_size].
        # And labels shape [batch_size * seq_len].
        # We'll reshape:
        batch_size, seq_len, vocab_size = logits.shape
        loss = criterion(
            logits.view(batch_size * seq_len, vocab_size),
            labels.view(batch_size * seq_len)
        )

        loss = loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch}, step {step}, loss = {loss.item()}")

torch.save(model.state_dict(), "finetuned_gpt_IT.pt")