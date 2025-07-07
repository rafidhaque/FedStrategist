# --- File: main.py ---

import argparse
import torch
from torch.utils.data import DataLoader
from utils import load_data, create_non_iid_partitions
from fl_core import Server, Client

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    trainset, testset = load_data()

    # Create non-IID partitions
    print(f"Creating {args.num_clients} non-IID partitions with beta={args.beta}...")
    client_datasets = create_non_iid_partitions(trainset, args.num_clients, args.beta)

    # Initialize server and clients
    server = Server(device)
    clients = [Client(client_id=i, dataset=client_datasets[i], device=device) for i in range(args.num_clients)]

    # Test loader for evaluation
    test_loader = DataLoader(testset, batch_size=64)

    # --- Federated Learning Loop ---
    for round_num in range(args.num_rounds):
        print(f"\n--- Round {round_num + 1}/{args.num_rounds} ---")
        
        # Get global model from server
        global_model_state = server.get_global_model_state()
        
        # Train clients and collect updates
        client_updates = []
        for client in clients:
            print(f"Training client {client.client_id}...")
            client.set_global_model(global_model_state)
            update = client.train(local_epochs=args.local_epochs)
            client_updates.append(update)
            
        # Server aggregates updates
        print(f"Server aggregating updates using '{args.agg_rule}'...")
        server.aggregate_updates(client_updates, args.agg_rule)
        
        # Evaluate global model
        accuracy = server.evaluate(test_loader)
        print(f"Round {round_num + 1} Global Model Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs for each client')
    parser.add_argument('--beta', type=float, default=0.5, help='Dirichlet distribution beta parameter for non-IID data')
    parser.add_argument('--agg_rule', type=str, default='fed_avg', choices=['fed_avg', 'median', 'krum'], help='Aggregation rule to use')
    
    args = parser.parse_args()
    main(args)