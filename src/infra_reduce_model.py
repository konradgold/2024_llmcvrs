import torch
from nanoGPT.sample_model import SampleMutableModel
from nanoGPT.prune_model import prune_model
import re
from finetune import finetune, CustomDataset
import copy
from openai import OpenAI
import random
from torch.utils.data import DataLoader
from datasets import load_dataset


class TextDataLoader:
    """
    A simple dataloader for text data that allows sampling random batches of texts.
    """
    def __init__(self, texts: list[str]):
        self.texts = texts

    def get_random_batch(self, batch_size):
        """
        Returns a random batch of texts.

        Args:
            batch_size (int): Number of texts to return in the batch.

        Returns:
            list: A random batch of texts.
        """
        if batch_size > len(self.texts):
            raise ValueError("Batch size exceeds the number of texts available.")
        return random.sample(self.texts, batch_size)


def evaluate_model(model, client, dataloader_validation):
    prompt = """
Are there any grammatical errors?
Is the text comprehensible and coherent? 
Provide a score from 1-5 where 5 means flawless and 1 means incomprehensible.
TEXT: {text}

Return only this:
Score: X
    """
    sm_model = SampleMutableModel()
    sm_model.model = model
    sm_model.max_new_tokens = 100
    sm_model.model.eval()
    judgement_scores = []
    with torch.no_grad():
        batch = dataloader_validation.get_random_batch(5)
        output = sm_model.generate_output(batch)
        for text in output:
            filled_prompt = prompt.format(text=text)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert language evaluator. Your task is to evaluate the grammatical accuracy and overall comprehensibility of the following text. Ignore the content or creativity of the text. Focus only on the grammar and coherence. Respect the output format that will be given to you."},
                    {"role": "user", "content": filled_prompt},
                ],
                stream=False,
            )
            judgement_score = response.choices[0].message.content if response.choices is not None else "0"
            judgement_scores.append(int(re.search(r'\d+', judgement_score)[0] if re.search(r'\d+', judgement_score) is not None else "0"))
    return sum(judgement_scores) / len(judgement_scores)
    

def objective_func(model, X, finetune_bool=False, model_orig=None):
    """
    model:    your underlying model that can be "reduced"
    X:        tensor of shape [N, 12], each row is a candidate.
    Returns:  tensor of shape [N, 1] with objective values
    """
    # We'll store results in a Python list, then stack into a torch.Tensor
    results = []
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    best_model = copy.deepcopy(model)
    for x_row in X:
        # Convert to Python float list if needed
        x_list = x_row.tolist()  # [12 floats]
        

        # 1) Modify (reduce) your model in-place
        model = prune_model(model, x_list)

        model.train()  # make sure it's in training mode
        dataloader_train = TextDataLoader(load_dataset("mintujupally/ROCStories", split="train")["text"])

        dataset = CustomDataset(dataloader_train.get_random_batch(200))
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        if finetune_bool:
            model = finetune(model, model_orig, dataloader, 2)
            print("Model finetuned")

        dataloader_validation = TextDataLoader(load_dataset("mintujupally/ROCStories", split="test")["text"])

        model.eval()
        # 2) Evaluate the model
        y_value = evaluate_model(model, client, dataloader_validation)  #  a float (or maybe a python scalar)
        print(f"Evaluated model with score {y_value}")
        # 3) Append
        if len(results) == 0:
            best_model = copy.deepcopy(model)
        elif y_value > max(results):
            best_model = copy.deepcopy(model)
        results.append(y_value)
        
    return torch.tensor([results], dtype=torch.float32), best_model
