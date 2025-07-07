# --- File: aggregation.py ---

import torch
from collections import OrderedDict

def fed_avg(updates):
    """
    Performs Federated Averaging.
    Args:
        updates (list of OrderedDict): A list of model state_dicts from clients.
    Returns:
        OrderedDict: The new global model state_dict.
    """
    if not updates:
        return OrderedDict()

    # Initialize a new global model with zeros
    agg_update = OrderedDict()
    for key in updates[0].keys():
        agg_update[key] = torch.zeros_like(updates[0][key])

    # Sum up all the updates
    for update in updates:
        for key in update.keys():
            agg_update[key] += update[key]

    # Average the updates
    num_updates = len(updates)
    for key in agg_update.keys():
        agg_update[key] /= num_updates
        
    return agg_update

def coordinate_wise_median(updates):
    """
    Performs coordinate-wise median aggregation. Robust to outliers.
    Args:
        updates (list of OrderedDict): A list of model state_dicts from clients.
    Returns:
        OrderedDict: The new global model state_dict.
    """
    if not updates:
        return OrderedDict()

    # Stack all updates for each parameter
    stacked_updates = OrderedDict()
    for key in updates[0].keys():
        # Stack the tensors from all clients for the current key
        stacked_updates[key] = torch.stack([update[key] for update in updates], dim=0)

    # Compute the median along the client dimension
    median_update = OrderedDict()
    for key, stacked_tensor in stacked_updates.items():
        median_update[key] = torch.median(stacked_tensor, dim=0).values
    
    print("Aggregation complete using Coordinate-wise Median.")
    return median_update


def krum(updates, num_malicious=1, num_to_select=1):
    """
    Performs Krum aggregation. Selects the `num_to_select` updates that are closest
    to their k nearest neighbors. Assumes `n > 2f + 2` where f is num_malicious.
    Args:
        updates (list of OrderedDict): List of model state_dicts from clients.
        num_malicious (int): The number of assumed malicious clients (f).
        num_to_select (int): The number of "good" clients to average (k).
    Returns:
        OrderedDict: The new global model state_dict.
    """
    num_clients = len(updates)
    if num_clients <= 2 * num_malicious + 2:
        print(f"Warning: Krum requires n > 2f + 2. Got n={num_clients}, f={num_malicious}. Falling back to FedAvg.")
        return fed_avg(updates)

    # Step 1: Flatten each client's update into a single vector
    flattened_updates = []
    for update in updates:
        flattened_updates.append(torch.cat([param.view(-1) for param in update.values()]))

    # Step 2: Calculate pairwise squared Euclidean distances
    distances = torch.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i, num_clients):
            dist = torch.norm(flattened_updates[i] - flattened_updates[j], p=2) ** 2
            distances[i, j] = dist
            distances[j, i] = dist

    # Step 3: For each client, find the sum of distances to its k nearest neighbors
    # k = n - f - 2
    num_neighbors = num_clients - num_malicious - 2
    scores = torch.zeros(num_clients)
    for i in range(num_clients):
        # Sort distances to find the nearest neighbors
        sorted_dists, _ = torch.sort(distances[i])
        # Sum the distances to the k nearest neighbors (excluding self, which is index 0)
        scores[i] = torch.sum(sorted_dists[1:num_neighbors+1])

    # Step 4: Select the client(s) with the lowest scores
    _, top_indices = torch.topk(scores, k=num_to_select, largest=False)

    # Step 5: Aggregate the selected updates (using FedAvg on the subset)
    selected_updates = [updates[i] for i in top_indices]
    
    print(f"Krum: Selected updates from client(s) {top_indices.tolist()}.")
    return fed_avg(selected_updates)