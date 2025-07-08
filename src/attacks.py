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


def stealth_poisoning_attack(malicious_update, global_model_state, target_norm):
    """
    Implements a stealthier version of model poisoning.
    The attack crafts a malicious update and then scales it so its L2 norm
    matches a target norm (e.g., the average norm of benign clients),
    making it harder to detect via simple norm-based outlier detection.

    Args:
        malicious_update (OrderedDict): The state_dict from a malicious client's initial update.
        global_model_state (OrderedDict): The state_dict of the global model.
        target_norm (float): The desired L2 norm for the final malicious update vector.

    Returns:
        OrderedDict: The stealthy malicious model state_dict.
    """
    # Calculate the direction of the malicious update
    malicious_direction_vector = []
    for key in malicious_update:
        malicious_direction_vector.append((malicious_update[key] - global_model_state[key]).view(-1))
    malicious_direction_vector = torch.cat(malicious_direction_vector)
    
    # Normalize the malicious direction to a unit vector
    direction_norm = torch.norm(malicious_direction_vector, p=2)
    if direction_norm == 0: # Avoid division by zero
        return malicious_update # Return original if no direction
        
    normalized_direction = malicious_direction_vector / direction_norm
    
    # Scale this unit vector by the target norm
    stealth_direction_vector = target_norm * normalized_direction
    
    # Reconstruct the state_dict from the scaled vector
    stealth_update = global_model_state.copy()
    current_pos = 0
    for key in stealth_update:
        param_shape = stealth_update[key].shape
        param_size = stealth_update[key].numel()
        # Take the corresponding slice from the stealth vector and reshape it
        param_update = stealth_direction_vector[current_pos:current_pos + param_size].view(param_shape)
        stealth_update[key] += param_update # Add the scaled update to the global model
        current_pos += param_size
        
    return stealth_update


# --- Placeholder for future, more sophisticated attacks ---
def adaptive_attack_krum():
    pass