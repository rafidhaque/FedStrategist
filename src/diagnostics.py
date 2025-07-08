# --- File: diagnostics.py ---

import torch
import torch.nn.functional as F

def get_update_norms(updates):
    """Calculates the L2 norm of each client update."""
    norms = []
    for update in updates:
        # Flatten all parameters into a single vector for the norm calculation
        flat_update = torch.cat([param.view(-1) for param in update.values()])
        norms.append(torch.norm(flat_update, p=2))
    return torch.stack(norms)

def get_pairwise_cosine_similarity(updates):
    """Calculates the average pairwise cosine similarity between all client updates."""
    if len(updates) < 2:
        return torch.tensor(1.0) # If only one client, similarity is perfect

    # Flatten updates
    flattened_updates = []
    for update in updates:
        flattened_updates.append(torch.cat([param.view(-1) for param in update.values()]))
    
    flattened_updates = torch.stack(flattened_updates) # Shape: [num_clients, num_params]
    
    # Normalize each client's update vector to unit length
    normalized_updates = F.normalize(flattened_updates, p=2, dim=1)
    
    # Compute the cosine similarity matrix (dot product of normalized vectors)
    similarity_matrix = torch.matmul(normalized_updates, normalized_updates.T)
    
    # We only need the upper triangle of the matrix (excluding the diagonal)
    # as the matrix is symmetric and the diagonal is all 1s.
    upper_tri_indices = torch.triu_indices(len(updates), len(updates), offset=1)
    pairwise_similarities = similarity_matrix[upper_tri_indices[0], upper_tri_indices[1]]
    
    return torch.mean(pairwise_similarities)

# Note: The 'delta_accuracy' metric will be computed inside the Server class,
# as it requires access to the model's performance before and after aggregation.