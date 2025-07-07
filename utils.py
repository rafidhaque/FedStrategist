# --- File: utils.py ---

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_data(data_dir='./data'):
    """
    Loads the CIFAR-10 dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    
    return trainset, testset

def create_non_iid_partitions(dataset, num_clients, beta=0.5):
    """
    Partitions a dataset into non-IID subsets using a Dirichlet distribution.
    The beta parameter controls the degree of non-IID-ness.
    A small beta (e.g., 0.1) creates highly non-IID partitions.
    A large beta (e.g., 10) creates more IID-like partitions.
    
    Returns a list of Subset objects, one for each client.
    """
    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    
    # Create a list of indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Store client data indices
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = class_indices[k]
        np.random.shuffle(idx_k)
        
        # Proportions for clients for this class from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        
        # Ensure proportions are not too skewed to avoid some clients getting 0 data
        proportions = np.array([p * (len(idx_j) < len(dataset) / num_clients) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        
        # Distribute indices based on proportions
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split and assign to clients
        client_indices_k = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i].extend(client_indices_k[i])
            
    # Create PyTorch Subset objects for each client
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets