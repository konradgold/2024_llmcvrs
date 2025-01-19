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

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/knowledge.json", "r") as f:
    samples = json.load(f)

print(len(samples))

documents = [s["sentence"] + s["object_ground"] for s in samples]


# Prepare embeddings
embeddings = model.encode(documents, show_progress_bar=False)

# Train BERTopic
topic_model = topic_model.fit(documents, embeddings)

# Run the visualization with the original embeddings
fig = topic_model.visualize_document_datamap(documents, embeddings=embeddings)

## Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
#reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

fig.savefig("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/output/datamap-full_model.png", bbox_inches="tight")