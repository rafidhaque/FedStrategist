#!/bin/bash

echo "--- Starting Experiment 1: Benchmarking FedStrategist ---"

# --- Scenario 1: Moderate Heterogeneity, Moderate Attack ---
BETA=0.5
MALICIOUS=5
CLIENTS=20
ROUNDS=50
SEED=42

echo "\n--- Running Scenario 1: beta=$BETA, malicious=$MALICIOUS ---"

python main.py --agg_rule=adaptive --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED
python main.py --agg_rule=fed_avg --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED
python main.py --agg_rule=median --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED
python main.py --agg_rule=krum --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED


# --- Scenario 2: High Heterogeneity (Non-IID), Moderate Attack ---
BETA=0.1
MALICIOUS=5

echo "\n--- Running Scenario 2: beta=$BETA, malicious=$MALICIOUS ---"

python main.py --agg_rule=adaptive --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED
python main.py --agg_rule=fed_avg --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED
python main.py --agg_rule=median --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED
python main.py --agg_rule=krum --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED


echo "\n--- Experiment 1 Complete ---"