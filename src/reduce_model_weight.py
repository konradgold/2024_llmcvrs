from timm import data
import torch
from nanoGPT.sample_model import SampleMutableModel
from nanoGPT.prune_model import prune_model
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
import re
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from finetune import finetune, dataloader
import copy
from openai import OpenAI
import random
from datasets import load_dataset

ds = load_dataset("mintujupally/ROCStories")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sm_model = SampleMutableModel()
model = sm_model.model
state_dict = torch.load('finetuned_gpt_IT.pt', weights_only=False)
model.load_state_dict(state_dict, strict=True)
model_orig = copy.deepcopy(model)
model.to(device)
model_orig.to(device)

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
You are an expert language evaluator. Your task is to evaluate the grammatical accuracy and overall comprehensibility of the following text. Ignore the content or creativity of the text. Focus only on the following:

Are there any grammatical errors?
Is the text comprehensible and coherent? Provide a score from 1-5 where 5 means flawless and 1 means incomprehensible.
TEXT: {text}

Return only this:
Score: X
    """
    sm_model = SampleMutableModel()
    sm_model.model.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    judgement_scores = []
    with torch.no_grad():
        batch = dataloader_validation.get_random_batch(5)
        output = sm_model.generate_output(batch)
        for text in output:
            filled_prompt = prompt.format(text=text)
            response = client.completions.create(
            model="gemma-2-9b-it",
            prompt=filled_prompt,
            )
            judgement_score = response.choices[0].text if response.choices is not None else "0"
            print(judgement_score)
            judgement_scores.append(int(re.search(r'\d+', judgement_score) if re.search(r'\d+', judgement_score) is not None else 0))
    return sum(judgement_scores) / len(judgement_scores)
    

def objective_func(model, X):
    """
    model:    your underlying model that can be "reduced"
    X:        tensor of shape [N, 12], each row is a candidate.
    Returns:  tensor of shape [N, 1] with objective values
    """
    # We'll store results in a Python list, then stack into a torch.Tensor
    results = []
    client = OpenAI(base_url="http://127.0.0.1:1234", api_key="lm-studio")
    for x_row in X:
        # Convert to Python float list if needed
        x_list = x_row.tolist()  # [12 floats]

        # 1) Modify (reduce) your model in-place
        model = prune_model(model, x_list)

        model.train()  # make sure it's in training mode
        model = finetune(model, model_orig, dataloader, 5)

        # 2) Evaluate the model
        y_value = evaluate_model(model, client)  # a float (or maybe a python scalar)
        
        # 3) Append
        results.append([y_value])

    return torch.tensor(results, dtype=torch.float32)

client = OpenAI(base_url="http://127.0.0.1:1234", api_key="lm-studio")
dataloader_validation = TextDataLoader(load_dataset("mintujupally/ROCStories", split="train")["text"])
evaluate_model(model, client, dataloader)
exit()

## Nr 1: measure language model performance
NUM_INIT = 5
bounds = torch.stack([torch.zeros(12), torch.ones(12)*0.2])

X_init = torch.rand(NUM_INIT, 12)  # random points in [0,1]
Y_init = objective_func(model, X_init)       # evaluate your expensive function

gp = SingleTaskGP(X_init, Y_init)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# Fit the GP hyperparameters
fit_gpytorch_model(mll)

N_ITER = 10  # how many BO steps you want

for i in range(N_ITER):
    # Create the acquisition function
    # If we're *maximizing* f, "best_f" is the best Y we have so far
    best_f = Y_init.max().item()
    EI = ExpectedImprovement(model=gp, best_f=best_f)
    
    # Optimize the acquisition function over the bounds
    #   bounds shape: [2, dim], where dim=12
    #   We'll request a single candidate (q=1)
    candidate, acq_value = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=1,
        num_restarts=5,       # random restarts for global opt
        raw_samples=20,       # samples for initialization
    )
    
    # 'candidate' now is shape [1, 12], the recommended next point
    # Evaluate it
    new_y = objective_func(model, candidate)
    
    # Augment our data
    X_init = torch.cat([X_init, candidate], dim=0)  # now shape [(5 + i+1), 12]
    Y_init = torch.cat([Y_init, new_y], dim=0)      # shape [(5 + i+1), 1]
    
    # Re-fit the GP to the new data
    gp = SingleTaskGP(X_init, Y_init)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    print(f"Iteration {i+1}: best Y so far = {Y_init.max().item():.3f}")