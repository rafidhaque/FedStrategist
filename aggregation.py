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

# --- Placeholder for future rules ---
def coordinate_wise_median(updates):
    # [TODO at Camp 1, Stage 2]
    print("WARNING: coordinate_wise_median not yet implemented. Using fed_avg.")
    return fed_avg(updates)

def krum(updates):
    # [TODO at Camp 1, Stage 2]
    print("WARNING: krum not yet implemented. Using fed_avg.")
    return fed_avg(updates)