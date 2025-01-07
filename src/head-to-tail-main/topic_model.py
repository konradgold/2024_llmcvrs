import os
import json

## 1. get data

data_directory = '/Users/konradgoldenbaum/Developement/LLMCVRS/src/head-to-tail-main/data'
data_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

data = {"head": [], "torso":[], "tail": []}
for file in data_files:
    with open(file, 'r') as f:
        data["head"].append(json.load(f)["head"])
        data["torso"].append(json.load(f)["torso"])
        data["tail"].append(json.load(f)["tail"])

## 2. preprocess data
## 3. get topics
## 4. save topics