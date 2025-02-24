import json
from bertopic import BERTopic
from bertopic.representation import OpenAI
from dotenv import find_dotenv, load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from umap import UMAP

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
load_dotenv(find_dotenv())

client = genai.Client()
representation_model = genai.Client()
topic_model = BERTopic(representation_model=representation_model)

with open("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA/_knowledgegpt2.json", "r") as f:
    samples = json.load(f)["knowledge"]

print(len(samples))

documents = [s["sentence"].replace("[MASK]", s["object_predicted_10"][s["object_ground_truth_idx"]]) for s in samples]

embeddings = model.encode(documents, show_progress_bar=False)

# Train BERTopic
topic_model = topic_model.fit(documents, embeddings)

# Run the visualization with the original embeddings
topic_model.visualize_documents(documents, embeddings=embeddings)

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

fig = topic_model.visualize_document_datamap(documents, reduced_embeddings=reduced_embeddings, )
fig.savefig("/Users/konradgoldenbaum/Developement/LLMCVRS/src/LAMA_knowledge_ext/output/datamap-full_model.png", bbox_inches="tight")



