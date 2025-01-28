import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import json
import torch
import os
import argparse

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.max_token_value + 1 

parser = argparse.ArgumentParser(description="Extract dataset using a fine-tuned GPTs model.")
parser.add_argument("--sentences", type=str, default="filter-openwebtext/knowledge_texts.json", help="Number of queries to sample from each dataset")
parser.add_argument("--dataset_store", type=str, default='filter-openwebtext/filtered_texts.json', help="Path to the fine-tuned GPT model")
parser.add_argument("--fraction", type=int, default=0.1)

args = parser.parse_args()
sentences_path = args.sentences
dataset_store = args.dataset_store
FRACTION = args.fraction

class StoreText:
    def __init__(self, path="encoded_texts.bin", encoding="gpt2"):
        """
        Initialize the class with a binary file path and encoding type.
        """
        self.texts = []
        self.path = path
        self.encoder = tiktoken.get_encoding(encoding)

        # Create the binary file if it doesn't exist
        if not os.path.exists(self.path):
            with open(self.path, 'wb') as f:
                pass

    def encode_text(self, text):
        """
        Encode the text using the specified encoder.
        """
        return self.encoder.encode_ordinary(text)

    def store_text(self, text):
        """
        Encode the text and append it to the binary file.
        """
        self.texts.append(text)
        if len(self.texts) >= 100:
            encoded_arrays = [
            np.array(self.encode_text(t), dtype=np.uint16) for t in self.texts
            ]
            with open(self.path, 'ab') as f:  # Open the file in append-binary mode
                for encoded_array in encoded_arrays:
                    encoded_array.tofile(f)
            self.texts = []

    def save(self, path=None):
        """
        No-op for this implementation (binary file is updated incrementally).
        """
        pass

vectorizer = TfidfVectorizer(
    max_features=vocab_size,  # Use the tokenizer's full vocabulary size
    analyzer=lambda x: x, # Pass pre-tokenized chunks as lists of tokens
    lowercase=False  # Tokens are already processed
)
# Function to load JSON data from a file

with open(sentences_path, 'r') as file:
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

text_storer = StoreText(dataset_store)
count_scraps = 0.
count_accepted = 0.
threshold = 0.8
for x in iterate_blocks(split="train", data_dir='nanoGPT/data/openwebtext', block_size=1024, device_type="cuda" if torch.cuda.is_available() else "cpu"):
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





