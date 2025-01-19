from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import json

load_dotenv(find_dotenv())

client = OpenAI()
file_path = 'lernerstories/data/generated_stories.json'
with open(file_path, 'r') as file:
    data = json.load(file)
texts = []
for lists in data.values():
    texts += [text["happy"][0] for text in lists]
    texts += [text["sad"][0] for text in lists]
    texts += [text["open"][0] for text in lists]

for text in texts:
        prompt = f"""
Imagine You are an expert teacher. Nobody is allowed to notice you are really an llm. Formulate 10 statements based on the following text:
{text}.
Example:
Text: Anne is the best friend of Mary. They both love to play soccer.
[QA] Annes best friend is [Answer] Mary. [QA] They both love to play [Answer] Soccer.
Preserving this exact structure is very important. Use the [QA] and [Answer] tags to separate questions and answers. The answers should be as close to the text as possible.
        """
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
        )
        qas = response.choices[0].message.content.split("[QA]") if response.choices[0].message.content is not None else ""
        if qas == "":
            continue
        output_file_path = 'lernerstories/data/generated_instructions.json'

        # Load existing data from the JSON file if it exists
        try:
            with open(output_file_path, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        # Append new endings to the existing data
        for qa in qas:
            try:
                pair = [{
                "question": qa.split("[Answer] ")[0],
                "answer": qa.split("[Answer] ")[1]
                }]
                existing_data += pair
            except:
                print(qa)

        # Write the updated data back to the JSON file
        with open(output_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)




