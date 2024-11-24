import json
from bertopic import BERTopic
from bertopic.representation import OpenAI
from dotenv import find_dotenv, load_dotenv
import openai
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
load_dotenv(find_dotenv())

client = openai.OpenAI()
representation_model = OpenAI(client, model="gpt-3.5-turbo", chat=True)
topic_model = BERTopic(representation_model=representation_model)

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA/_knowledgegpt2.json", "r") as f:
    samples = json.load(f)["knowledge"]

print(len(samples))

documents = [s["sentence"].replace("[MASK]", s["object_predicted_10"][s["object_ground_truth_idx"]]) for s in samples]

topics, probs = topic_model.fit_transform(documents=documents)

print(topics)
print(topic_model.get_topic_info())

