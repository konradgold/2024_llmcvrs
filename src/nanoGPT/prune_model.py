import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_neuron_pair_importance(c_fc_weight):
    """
    compute neuron pair importance scores (Maximum Absolute Weight)
    Args:
    - c_fc_weight: Weight matrix from the c_fc layer.
    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """
    importance_scores = torch.max(torch.abs(c_fc_weight), dim=1).values + torch.abs(torch.min(c_fc_weight, dim=1).values)
    return importance_scores

def compute_activation_importance(c_fc_activation):
    """
    compute neuron pair importance scores (Maximum Absolute Weight)
    Args:
    - c_fc_weight: Weight matrix from the c_fc layer.
    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """
    c_fc_activation = c_fc_activation.float()
    importance_scores = torch.std(c_fc_activation, dim=0)
    return importance_scores

def prune_neuron_pairs(mlp, prune_percent, activation_based=False):
    """
    Reduces the dimensions of the **gate_proj**,**up_proj**, **down_proj**
    layers removing the least important neurons.
    Args:
    - mlp: Layers to prune.
    - prune_percent: Percentage of neurons to prune.
    Returns:
    - new_gate_proj, new_up_proj, new_down_proj: New pruned layers.
    - k: New intermediate size.
    """
    # Extract weights from MLP layers

    # Compute importance scores
    if activation_based:
        importance_scores = compute_activation_importance(mlp.c_fc_activations)
    else:
        importance_scores = compute_neuron_pair_importance(mlp.c_fc.weight.data.float())
    assert importance_scores.size(0) == mlp.c_fc.weight.data.size(0)
    original_intermediate_size = mlp.c_fc.weight.data.float().size(0)
    
    # Calculate neurons to keep
    num_neuron_pairs_to_prune = min(int(prune_percent * original_intermediate_size),
                                   original_intermediate_size - 1)
    k = original_intermediate_size - num_neuron_pairs_to_prune
    
    # Validation check
    if k <= 0:
        raise ValueError(f"Invalid number of neuron pairs to keep: {k}")
    
    # Select neurons to keep
    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values
    
    # Create and populate new layers
    new_c_fc = nn.Linear(mlp.c_fc.in_features, k, bias=False).to(device)
    new_c_proj = nn.Linear(k, mlp.c_proj.out_features, bias=False).to(device)
    
    # Copy selected weights
    new_c_fc.weight.data = mlp.c_fc.weight.data[indices_to_keep, :]
    new_c_proj.weight.data = mlp.c_proj.weight.data[:, indices_to_keep]
    
    return new_c_fc, new_c_proj, k


def prune_model(model, prune_percent, activation_based=False):
    new_intermediate_size = None

    for idx, layer in enumerate(model.transformer.h):
        mlp = layer.mlp
        if isinstance(prune_percent, list) and len(prune_percent) == len(model.transformer.h):
            pp = prune_percent[idx]
        else:
            pp = prune_percent
        new_c_fc, new_c_proj, new_size = prune_neuron_pairs(
            mlp, pp, activation_based)

        new_c_fc.to(device)
            
        mlp.c_fc = new_c_fc
        mlp.c_proj = new_c_proj
        
        if new_intermediate_size is None:
            new_intermediate_size = new_size
    return model