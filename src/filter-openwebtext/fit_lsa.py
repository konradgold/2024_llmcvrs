import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import json
import torch
import os

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.max_token_value + 1 

class StoreText:
    def __init__(self, path="filtered_texts.json"):
        self.texts = []
        self.path = path

    def store_text(self, text):
        self.texts.append(text)
        if len(self.texts) % 100 == 0:
            self.save(self.path)

    def save(self, path=None):
        if path is None:
            path = self.path
        with open(path, 'w') as f:
            json.dump(self.texts, f)

vectorizer = TfidfVectorizer(
    max_features=vocab_size,  # Use the tokenizer's full vocabulary size
    analyzer=lambda x: x, # Pass pre-tokenized chunks as lists of tokens
    lowercase=False  # Tokens are already processed
)

file_path = "filter-openwebtext/knowledge_texts.json"
# Function to load JSON data from a file

with open(file_path, 'r') as file:
    data = json.load(file)

fit_set = [enc.encode(s) for s in data]

vectorizer.fit(fit_set)

tfidf_matrix = vectorizer.transform(fit_set)

# Apply LSA (Truncated SVD)
lsa_model = TruncatedSVD(n_components=15, random_state=42)  # Adjust components
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

def iterate_blocks(split, data_dir, block_size, device_type):
    device = torch.device(device_type)
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    for i in range(0, len(data), block_size):
        x = torch.from_numpy((data[i:i+block_size]).astype(np.int64))
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        yield x

text_storer = StoreText()
count_scraps = 0.
count_accepted = 0.
threshold = 0.8
FRACTION = 0.4
for x in iterate_blocks(split="train", data_dir='nanoGPT/data/openwebtext', block_size=100, device_type='cpu'):
    count_scraps += 1.
    if count_accepted/count_scraps > FRACTION + 3e-2:
        threshold += 1e-3
    if count_accepted/count_scraps < FRACTION - 3e-2:
        threshold -= 1e-3
    X_new = vectorizer.transform([x.tolist()])  # Uses the same vocabulary
    X_new_lsa = lsa_model.transform(X_new) 
    similarity = cosine_similarity(X_new_lsa, lsa_matrix)
    max_similarity = np.max(similarity[0])
    if max_similarity > threshold:
        count_accepted += 1.
        text = enc.decode(x.tolist())
        text_storer.store_text(text)
        print(f"Current Fraction: {count_accepted/count_scraps:.2f}", end='\r')
print("")
text_storer.save()





