# --- File: main.py (Final Version with Logging) ---

import argparse
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import load_data, create_non_iid_partitions
from fl_core import Server, Client, MaliciousClient
from diagnostics import get_update_norms

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Set run name for logging
    run_name = f"rule={args.agg_rule}_clients={args.num_clients}_malicious={args.num_malicious}_beta={args.beta}_seed={args.seed}"
    log_path = os.path.join(args.results_dir, f"{run_name}.csv")
    print(f"Logging results to: {log_path}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Data and Model Setup ---
    trainset, testset = load_data()
    client_datasets = create_non_iid_partitions(trainset, args.num_clients, args.beta)
    test_loader = DataLoader(testset, batch_size=128)

    # --- Initialize Server and Clients ---
    server = Server(device, args.num_clients)
    clients = []
    # Benign clients
    for i in range(args.num_clients - args.num_malicious):
        clients.append(Client(client_id=i, dataset=client_datasets[i], device=device))
    # Malicious clients
    for i in range(args.num_clients - args.num_malicious, args.num_clients):
        clients.append(MaliciousClient(client_id=i, dataset=client_datasets[i], device=device, attack_type=args.attack_type))

    print(f"Initialized {len(clients)} total clients ({args.num_malicious} malicious).")

    # --- Logging Setup ---
    results_log = []
    
    # --- Federated Learning Loop ---
    for round_num in range(args.num_rounds):
        print(f"\n--- Round {round_num + 1}/{args.num_rounds} ---")
        
        global_model_state = server.get_global_model_state()
        
        # --- Separate benign and malicious training ---
        benign_updates = []
        # First, process benign clients to establish a baseline
        for client in clients:
            if not isinstance(client, MaliciousClient):
                client.set_global_model(global_model_state)
                update = client.train(local_epochs=args.local_epochs)
                benign_updates.append(update)
        
        # Calculate the average norm of benign updates for the stealth attack
        avg_benign_norm = torch.mean(get_update_norms(benign_updates)).item() if benign_updates else 0

        malicious_updates = []
        for client in clients:
             if isinstance(client, MaliciousClient):
                update = client.generate_malicious_update(global_model_state, args.local_epochs, avg_benign_norm)
                malicious_updates.append(update)

        # Combine all updates for aggregation and diagnostics
        client_updates = benign_updates + malicious_updates
        
        pre_agg_accuracy = server.evaluate(test_loader)

        current_state_vector = server.compute_state_vector(client_updates)
        print(f"State Vector S_t: {current_state_vector.cpu().numpy().round(4)}")
        
        chosen_rule_str = args.agg_rule
        action_idx = -1 # Default for non-adaptive
        if args.agg_rule == 'adaptive':
            action_idx = server.bandit.choose_action(current_state_vector.cpu().numpy())
            chosen_rule_str = server.agg_rules[action_idx]
            print(f"Bandit chose action: {chosen_rule_str}")
        
        server.aggregate_updates(client_updates, chosen_rule_str, num_malicious=args.num_malicious)
        
        post_agg_accuracy = server.evaluate(test_loader)
        print(f"Round {round_num + 1} Global Model Accuracy: {post_agg_accuracy:.2f}%")
        
        reward = 0.0
        if args.agg_rule == 'adaptive':
            delta_accuracy = post_agg_accuracy - pre_agg_accuracy
            reward = server.calculate_reward(delta_accuracy, chosen_rule_str, args.lambda_cost)
            print(f"  Delta Acc: {delta_accuracy:.2f}, Cost: {server.agg_costs.get(chosen_rule_str, 0):.2f}, Reward: {reward:.4f}")
            server.bandit.update(action_idx, current_state_vector.cpu().numpy(), reward)

        # Log results for this round
        round_data = {
            'round': round_num + 1,
            'agg_rule': chosen_rule_str,
            'accuracy': post_agg_accuracy,
            'reward': reward,
            'state_norm_var': current_state_vector[0].item(),
            'state_cosine_sim': current_state_vector[1].item(),
            'state_mean_norm': current_state_vector[2].item(),
        }
        results_log.append(round_data)

    # --- Save final results ---
    df = pd.DataFrame(results_log)
    df.to_csv(log_path, index=False)
    print(f"\nResults saved to {log_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.5, help='Controls non-IIDness')
    parser.add_argument('--agg_rule', type=str, default='adaptive', choices=['fed_avg', 'median', 'krum', 'adaptive'])
    parser.add_argument('--num_malicious', type=int, default=5, help='Number of malicious clients')
    parser.add_argument('--attack_type', type=str, default='standard', choices=['standard', 'stealth'], help='Type of malicious attack')
    parser.add_argument('--lambda_cost', type=float, default=0.5, help='Cost weight for bandit reward')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)