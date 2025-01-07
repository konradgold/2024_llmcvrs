from bertopic import BERTopic
from bertopic.representation import OpenAI
from dotenv import find_dotenv, load_dotenv
import openai
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
load_dotenv(find_dotenv())
client = openai.OpenAI()
representation_model = OpenAI(client, model="gpt-3.5-turbo", chat=True)
topic_model = BERTopic(representation_model=representation_model, nr_topics=10)

with open('head_to_tail_goodreads.json', 'r') as f:
    data = json.load(f)

qa = []
for d in data["head"]:
    if isinstance(d[3], list):
        d[3] = ", ".join(d[3])
    qa.append(f"{d[2]}; {d[3]}")

print(len(qa))


topics, _ = topic_model.fit_transform(qa)
assert len(topics) == len(qa)
for i, q in enumerate(qa):
    t = topic_model.get_topic(topics[i])
    data["head"][i].append(t)

with open('head_to_tail_goodreads_with_topics.json', 'w') as f:
    json.dump({"head": data["head"]}, f, indent=4)

print(topic_model.get_topic_info())