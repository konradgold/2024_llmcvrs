import json
import tiktoken
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import get_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10

class CustomDataset(Dataset):
    def __init__(self, texts, max_length=300):
        self.max_length = max_length
        self.enc = tiktoken.get_encoding("gpt2")
        self.examples = [self.tokenise(text) for text in texts]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def tokenise(self, text):
        encoded = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        encoded = encoded[:self.max_length] if len(encoded)>self.max_length else encoded
        attention = torch.ones((self.max_length), dtype=torch.long)
        attention[len(encoded):] = 0
        token_pad = torch.zeros((self.max_length), dtype=torch.long)
        token_pad[:len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        return token_pad


def finetune(model, model_orig, dataloader, epochs=10):
    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    if model_orig is not None:
        model_orig.eval() 
        for param in model_orig.parameters():
            param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=1e-1)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(dataloader)*epochs)
    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch.to(device)
            logits, loss = model(inputs[:-1], inputs[1:])
            if model_orig is not None:
                logits_orig, _ = model_orig(inputs[:-1], inputs[1:])
                kl_loss = torch.nn.functional.kl_div(
                    input=torch.nn.functional.softmax(logits, dim=-1),
                    target=torch.nn.functional.softmax(logits_orig, dim=-1),
                    reduction="batchmean"
                )
                loss += kl_loss * 0.01
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model

    