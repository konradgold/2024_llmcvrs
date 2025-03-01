import torch
from nanoGPT.sample_model import SampleMutableModel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
import copy
from datasets import load_dataset
from infra_reduce_model import objective_func
import argparse
import json

ds = load_dataset("mintujupally/ROCStories")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sm_model = SampleMutableModel()
model_orig = copy.deepcopy(sm_model.model)
model_orig.to(device)

parser = argparse.ArgumentParser(description='Reduce model weight using Bayesian Optimization.')
parser.add_argument('--n_iter', type=int, default=5, help='Number of BO steps')
parser.add_argument('--width', type=float, default=0.2, help='First number between 0 and 1')
parser.add_argument('--min', type=float, default=0.6, help='Second number between 0 and 1')
args = parser.parse_args()

width = args.width
min_reduction = args.min

print(min_reduction)

## Nr 1: measure language model performance
NUM_INIT = 5
bounds = torch.stack([torch.zeros(12), torch.ones(12)*width]) + min_reduction

bounds.to(device)

X_init = torch.rand(NUM_INIT, 12, dtype=torch.double)*width + min_reduction  # random points in [0,0.2]
X_init.to(device)
Y_init, model = objective_func(sm_model.model, X_init, model_orig=model_orig, finetune_bool=False)       # evaluate your expensive function
Y_init = Y_init.to(device)
torch.save(model, f"models/finetuned_gpt_{min_reduction}.pt")


gp = SingleTaskGP(X_init, Y_init)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# Fit the GP hyperparameters
fit_gpytorch_mll(mll)



N_ITER = args.n_iter
N_ITER = 5  # how many BO steps you want

for i in range(N_ITER):
    model = copy.deepcopy(model_orig)
    model.to(device)
    # Create the acquisition function
    # If we're *maximizing* f, "best_f" is the best Y we have so far
    best_f = Y_init.max().item()
    EI = LogExpectedImprovement(model=gp, best_f=best_f)
    
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
    new_y, model = objective_func(model, candidate, finetune_bool=True)
    new_y = new_y.to(device)
    
    # Augment our data
    X_init = torch.cat([X_init, candidate], dim=0)  # now shape [(5 + i+1), 12]
    if new_y > Y_init.max():
        torch.save(model, f"models/finetuned_gpt_{min_reduction}.pt")
        print("Saved model")

    # Store the results in a dictionary
        
    Y_init = torch.cat([Y_init, new_y], dim=0)      # shape [(5 + i+1), 1]
    
    # Re-fit the GP to the new data
    gp = SingleTaskGP(X_init, Y_init)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    print(f"Iteration {i+1}: best Y so far = {Y_init.max().item():.3f}")

results = {
        "X": X_init.tolist(),
        "Y": Y_init.tolist()
    }

    # Save the results to a JSON file
with open(f"results_mlp_act_real_activation_{min_reduction}.json", "w") as f:
    json.dump(results, f)
