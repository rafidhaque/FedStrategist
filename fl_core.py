# --- File: fl_core.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import SimpleCNN
from aggregation import fed_avg, coordinate_wise_median, krum
from collections import OrderedDict

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
        self.device = device
        self.aggregation_fn_map = {
            'fed_avg': fed_avg,
            'median': coordinate_wise_median,
            'krum': krum
        }

    def get_global_model_state(self):
        return self.global_model.state_dict()

    def aggregate_updates(self, client_updates, agg_rule):
        if agg_rule not in self.aggregation_fn_map:
            raise ValueError(f"Unknown aggregation rule: {agg_rule}")
        
        aggregation_fn = self.aggregation_fn_map[agg_rule]
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