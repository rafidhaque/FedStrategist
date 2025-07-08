#!/bin/bash

echo "--- Starting Experiment 2: Robustness to Stealth Attack ---"

# --- Scenario: Moderate Heterogeneity, Stealth Attack ---
BETA=0.5
MALICIOUS=5
CLIENTS=20
ROUNDS=50
SEED=123 # Use a different seed for a new set of runs

echo "\n--- Running Stealth Attack Scenario: beta=$BETA, malicious=$MALICIOUS ---"

python main.py --agg_rule=adaptive --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=stealth
python main.py --agg_rule=fed_avg --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=stealth
python main.py --agg_rule=median --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=stealth
python main.py --agg_rule=krum --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=stealth

echo "\n--- Experiment 2 Complete ---"