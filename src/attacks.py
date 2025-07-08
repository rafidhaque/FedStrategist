# --- File: attacks.py (Refactored) ---

import torch

def generate_malicious_direction(benign_update, global_model_state):
    """
    Helper function to calculate the intended malicious direction.
    This is based on the direction of a normally trained model.
    """
    with torch.no_grad():
        direction_vector = []
        for key in benign_update:
            direction_vector.append((benign_update[key] - global_model_state[key]).view(-1))
        return torch.cat(direction_vector)

def scale_and_reconstruct(direction_vector, target_norm, global_model_state):
    """
    Helper function to scale a direction vector to a target norm and
    reconstruct a state_dict.
    """
    with torch.no_grad():
        # Normalize the malicious direction to a unit vector
        direction_norm = torch.norm(direction_vector, p=2)
        if direction_norm == 0:
            return global_model_state # Return original if no direction

        normalized_direction = direction_vector / direction_norm
        
        # Scale this unit vector by the target norm
        scaled_direction_vector = target_norm * normalized_direction
        
        # Reconstruct the state_dict
        stealth_update = global_model_state.copy()
        current_pos = 0
        for key in stealth_update:
            param_shape = stealth_update[key].shape
            param_size = stealth_update[key].numel()
            param_update = scaled_direction_vector[current_pos:current_pos + param_size].view(param_shape)
            stealth_update[key] = global_model_state[key] + param_update
            current_pos += param_size
            
        return stealth_update

def standard_poisoning_attack(direction_vector, global_model_state, scale_factor=5.0):
    """
    The standard scaling attack.
    """
    target_norm = torch.norm(direction_vector, p=2) * scale_factor
    return scale_and_reconstruct(direction_vector, target_norm, global_model_state)
    
def stealth_poisoning_attack(direction_vector, global_model_state, target_norm):
    """
    The stealth scaling attack.
    """
    return scale_and_reconstruct(direction_vector, target_norm, global_model_state)