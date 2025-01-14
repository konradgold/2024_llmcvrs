import pandas as pd
from openai import OpenAI
import json

# Define the path to the CSV file
csv_file_path = 'lernerstories/data/ENGLISH_CERF_WORDS.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Remove rows where the first column contains only one letter or special symbols
df = df[~df.iloc[:, 0].str.match(r'^[A-Za-z]$|^[^A-Za-z0-9]+$')]

# Remove duplicates based on the first column, keeping the lexicographically smallest value in the second column
df = df.sort_values(by=df.columns[1]).drop_duplicates(subset=df.columns[0], keep='first')

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

for i in range(10):
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        filtered_df = df[df.iloc[:, 1] == level].sample(n=4).iloc[:, 0].values
        print(list(filtered_df))
        stories = {"seed": list(filtered_df)}
        for ending in ["happy", "sad", "open"]:
            prompt = f"""
Imagine You are an expert storyteller. Nobody is allowed to notice you are really an llm. Generate a story of 150 words that I can give to my students to practice english. They currently have a niveau of {level}. Use the following words:
{filtered_df}.
It should have a(n) {ending} ending. Only return the story, no commentary.
            """
            response = client.completions.create(
                model="gemma-2-9b-it",
                prompt=prompt,
            )
            response.choices[0].text.strip()
            stories.setdefault(ending, []).append(response.choices[0].text.strip())
        output_file_path = 'lernerstories/data/generated_stories.json'

        # Load existing data from the JSON file if it exists
        try:
            with open(output_file_path, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {}

        # Append new endings to the existing data
        existing_data.setdefault(level, []).append(stories)

        # Write the updated data back to the JSON file
        with open(output_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    break




