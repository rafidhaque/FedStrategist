# --- File: attacks.py ---

import torch

def local_model_poisoning_attack(benign_update, global_model_state, scale_factor=3.0):
    """
    Implements a simple version of the local model poisoning attack.
    The attack scales the difference between the global model and the client's
    (benign) update to create a malicious update that is in the same direction
    but much larger in magnitude.

    Args:
        benign_update (OrderedDict): The state_dict from a client's honest training.
        global_model_state (OrderedDict): The state_dict of the global model from the start of the round.
        scale_factor (float): How much to magnify the update direction.

    Returns:
        OrderedDict: The malicious model state_dict.
    """
    malicious_update = benign_update.copy()
    for key in malicious_update:
        # Calculate the update direction
        update_direction = benign_update[key] - global_model_state[key]
        
        # Scale the update direction to create the poisoned update
        poisoned_direction = scale_factor * update_direction
        
        # Add the poisoned direction to the original global model state
        malicious_update[key] = global_model_state[key] + poisoned_direction
        
    return malicious_update

# --- Placeholder for future, more sophisticated attacks ---
def adaptive_attack_krum():
    pass