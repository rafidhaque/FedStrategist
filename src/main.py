# --- File: main.py ---

import argparse
import torch
from torch.utils.data import DataLoader
from utils import load_data, create_non_iid_partitions
from fl_core import Server, Client, MaliciousClient


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
    server = Server(device, args.num_clients) # Pass num_clients
    
    clients = []
    # Benign clients
    for i in range(args.num_clients - args.num_malicious):
        clients.append(Client(client_id=i, dataset=client_datasets[i], device=device))
    # Malicious clients
    for i in range(args.num_clients - args.num_malicious, args.num_clients):
        clients.append(MaliciousClient(client_id=i, dataset=client_datasets[i], device=device))

    print(f"Initialized {len(clients)} total clients ({args.num_malicious} malicious).")

    test_loader = DataLoader(testset, batch_size=64)

    # --- Federated Learning Loop ---
    for round_num in range(args.num_rounds):
        print(f"\n--- Round {round_num + 1}/{args.num_rounds} ---")
        
        global_model_state = server.get_global_model_state()
        
        client_updates = []
        for client in clients:
            if isinstance(client, MaliciousClient):
                update = client.generate_malicious_update(global_model_state, local_epochs=args.local_epochs)
            else:
                client.set_global_model(global_model_state)
                update = client.train(local_epochs=args.local_epochs)
            client_updates.append(update)
            
        # Evaluate accuracy *before* aggregation to calculate delta
        pre_agg_accuracy = server.evaluate(test_loader)
        
        # --- COMMANDER LOGIC ---
        # Compute state vector based on the updates received
        # Note: We compute state before aggregation, but reward after
        current_state_vector = server.compute_state_vector(client_updates)
        # The state vector is on the GPU, but the bandit expects a numpy array.
        # We must move it to the CPU first before converting.
        current_state_numpy = current_state_vector.cpu().numpy()
        print(f"State Vector S_t: {current_state_numpy.round(4)}")
        
        chosen_rule_str = args.agg_rule
        if args.agg_rule == 'adaptive':
            # Ask the bandit to choose an action
            action_idx = server.bandit.choose_action(current_state_numpy)
            chosen_rule_str = server.agg_rules[action_idx]
            print(f"Bandit chose action: {chosen_rule_str}")
        
        # Server aggregates updates
        server.aggregate_updates(client_updates, chosen_rule_str, num_malicious=args.num_malicious)
        
        # Evaluate global model *after* aggregation
        post_agg_accuracy = server.evaluate(test_loader)
        print(f"Round {round_num + 1} Global Model Accuracy: {post_agg_accuracy:.2f}%")
        
        # --- UPDATE BANDIT ---
        if args.agg_rule == 'adaptive':
            delta_accuracy = post_agg_accuracy - pre_agg_accuracy
            reward = server.calculate_reward(delta_accuracy, chosen_rule_str, args.lambda_cost)
            print(f"  Delta Acc: {delta_accuracy:.2f}, Cost: {server.agg_costs.get(chosen_rule_str, 0):.2f}, Reward: {reward:.4f}")
            server.bandit.update(action_idx, current_state_numpy, reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    # ... existing args ...
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs for each client')
    parser.add_argument('--beta', type=float, default=0.5, help='Dirichlet distribution beta parameter for non-IID data')
    parser.add_argument('--agg_rule', type=str, default='fed_avg', choices=['fed_avg', 'median', 'krum', 'adaptive'], help='Aggregation rule to use')
    parser.add_argument('--num_malicious', type=int, default=0, help='Number of malicious clients')
    parser.add_argument('--lambda_cost', type=float, default=0.5, help='Weight of the cost in the reward function for the bandit')
    
    args = parser.parse_args()
    
    if args.num_malicious >= args.num_clients:
        raise ValueError("Number of malicious clients cannot be greater than or equal to the total number of clients.")
        
    main(args)
