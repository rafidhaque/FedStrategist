# --- File: fl_core.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import SimpleCNN
from aggregation import fed_avg, coordinate_wise_median, krum
from collections import OrderedDict
from attacks import local_model_poisoning_attack
from diagnostics import get_update_norms, get_pairwise_cosine_similarity

class Client:
    def __init__(self, client_id, dataset, device):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.model = SimpleCNN().to(self.device)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def set_global_model(self, global_model_state):
        self.model.load_state_dict(global_model_state)

    def train(self, local_epochs=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.model.train()
        for epoch in range(local_epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()

class Server:
    def __init__(self, device):
        self.global_model = SimpleCNN().to(device)
        self.previous_accuracy = 0.0
        self.device = device
        self.aggregation_fn_map = {
            'fed_avg': fed_avg,
            'median': coordinate_wise_median,
            'krum': krum
        }

    def get_global_model_state(self):
        return self.global_model.state_dict()

    def aggregate_updates(self, client_updates, agg_rule, num_malicious=0):
        if agg_rule not in self.aggregation_fn_map:
            raise ValueError(f"Unknown aggregation rule: {agg_rule}")
        
        aggregation_fn = self.aggregation_fn_map[agg_rule]
        
        if agg_rule == 'krum':
            aggregated_state_dict = aggregation_fn(client_updates, num_malicious=num_malicious)
        else:
            aggregated_state_dict = aggregation_fn(client_updates)
        
        self.global_model.load_state_dict(aggregated_state_dict)


    def evaluate(self, test_loader):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def compute_state_vector(self, client_updates, current_accuracy):
        """
        Computes the state vector S_t based on the latest client updates.
        
        Args:
            client_updates (list): A list of model state_dicts from clients.
            current_accuracy (float): The accuracy of the global model after aggregation.

        Returns:
            torch.Tensor: The state vector for the current round.
        """
        with torch.no_grad():
            # Metric 1: Variance of update norms
            update_norms = get_update_norms(client_updates)
            norm_variance = torch.var(update_norms).item()

            # Metric 2: Average pairwise cosine similarity
            avg_cosine_sim = get_pairwise_cosine_similarity(client_updates).item()

            # Metric 3: Change in global model accuracy
            delta_accuracy = current_accuracy - self.previous_accuracy
            
            # Update the stored accuracy for the next round
            self.previous_accuracy = current_accuracy
            
            # Combine metrics into a state vector
            # Note: The order here is important and must be consistent
            state_vector = torch.tensor([norm_variance, avg_cosine_sim, delta_accuracy], device=self.device)
            return state_vector


class MaliciousClient(Client):
    def __init__(self, client_id, dataset, device):
        super().__init__(client_id, dataset, device)
        self.attack_scale_factor = 5.0 # This can be tuned

    def train(self, local_epochs=1):
        # First, perform a benign training step to get a realistic update direction
        benign_update = super().train(local_epochs)
        
        # Get the global model state it started with
        # Note: In a real attack, the client would have stored this. Here we simulate.
        # For simplicity, we assume the benign_update's "starting point" is recoverable.
        # This is a simplification; more robust code would pass the global state in.
        # Let's refine this to be more explicit.
        
        # The 'train' method in the parent class should not be called directly
        # Instead, we will pass the global model state to the attack function
        pass

    def generate_malicious_update(self, global_model_state, local_epochs=1):
        """
        A dedicated method for generating the malicious update.
        """
        # Step 1: Perform a benign update to find a plausible direction
        self.set_global_model(global_model_state)
        benign_update = super().train(local_epochs) # The super().train() now returns the state dict
        
        # Step 2: Use the benign update and global state to craft the poisoned update
        malicious_update = local_model_poisoning_attack(
            benign_update,
            global_model_state,
            scale_factor=self.attack_scale_factor
        )
        
        return malicious_update

    