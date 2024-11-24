import json
import tiktoken

enc = tiktoken.get_encoding("gpt2")

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA/gpt/nano-gpt/vocab.json", "r") as f:
    samples = json.load(f)

new_vocab = {}

for i in range(50256):
    try:
        new_vocab[i] = enc.decode([i])
    except:
        print(i)
        print(enc.decode([i]))

count = 0
for term, i in samples.items():
    if term not in new_vocab.values():
        count += 1
print(count)