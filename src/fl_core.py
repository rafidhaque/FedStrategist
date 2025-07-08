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
from bandit import LinUCB
import numpy as np
from attacks import local_model_poisoning_attack, stealth_poisoning_attack


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
    
    def compute_state_vector(self, client_updates): # No longer needs accuracy
        """
        Computes the state vector S_t based on the latest client updates.
        The 'delta_accuracy' component is removed as it's part of the reward, not the state.
        """
        with torch.no_grad():
            update_norms = get_update_norms(client_updates)
            norm_variance = torch.var(update_norms).item()

            avg_cosine_sim = get_pairwise_cosine_similarity(client_updates).item()
            
            # New Metric: We can add the norm of the mean update as a third feature
            mean_update_norm = torch.norm(get_update_norms([fed_avg(client_updates)])).item()
            
            state_vector = torch.tensor([norm_variance, avg_cosine_sim, mean_update_norm], device=self.device)
            return state_vector

    
    def __init__(self, device, num_clients): # Add num_clients
        self.global_model = SimpleCNN().to(device)
        self.device = device
        self.aggregation_fn_map = {
            'fed_avg': fed_avg,
            'median': coordinate_wise_median,
            'krum': krum
        }
        self.agg_rules = list(self.aggregation_fn_map.keys()) # Ordered list of rules
        self.previous_accuracy = 0.0
        
        # Initialize the bandit agent
        # d=3 for our 3 state metrics, num_actions for our 3 rules
        self.bandit = LinUCB(num_actions=len(self.agg_rules), d=3, alpha=1.5)
        
        # Heuristic costs for each aggregation rule
        self.agg_costs = {'fed_avg': 0.1, 'median': 0.4, 'krum': 0.8}

    # Add a new method to the Server class to calculate reward
    def calculate_reward(self, delta_accuracy, chosen_rule, lambda_cost=0.5):
        """Calculates the reward for the bandit."""
        cost = self.agg_costs.get(chosen_rule, 0.0)
        reward = delta_accuracy - (lambda_cost * cost)
        return reward



class MaliciousClient(Client):
    def __init__(self, client_id, dataset, device, attack_type='standard'):
        super().__init__(client_id, dataset, device)
        self.attack_scale_factor = 5.0 # For the standard attack
        self.attack_type = attack_type

    def generate_malicious_update(self, global_model_state, local_epochs=1, benign_avg_norm=None):
        """
        Generates a malicious update based on the configured attack type.
        """
        # Step 1: Perform a benign update to find a plausible malicious direction
        self.set_global_model(global_model_state)
        initial_malicious_update = super().train(local_epochs=local_epochs)
        
        # Step 2: Craft the final attack
        if self.attack_type == 'stealth' and benign_avg_norm is not None:
            # Use the stealth attack that matches the norm of benign clients
            final_malicious_update = stealth_poisoning_attack(
                initial_malicious_update,
                global_model_state,
                target_norm=benign_avg_norm
            )
        else:
            # Default to the standard, naive scaling attack
            final_malicious_update = local_model_poisoning_attack(
                initial_malicious_update,
                global_model_state,
                scale_factor=self.attack_scale_factor
            )
            
        return final_malicious_update
    