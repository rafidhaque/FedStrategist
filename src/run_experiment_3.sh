#!/bin/bash

echo "--- Starting Experiment 3: Tuning FedStrategist's Risk Tolerance (lambda_cost) ---"

# --- Scenario: Moderate Heterogeneity, Stealth Attack ---
BETA=0.5
MALICIOUS=5
CLIENTS=20
ROUNDS=50
SEED=2024 # A new seed for this experiment
ATTACK="stealth"

echo -e "\n--- Running Stealth Attack Scenario: beta=$BETA, malicious=$MALICIOUS, attack=$ATTACK ---"

# --- Run FedStrategist with different lambda values ---
LAMBDAS=(0.1 0.5 1.0 2.0)
for LAMBDA in "${LAMBDAS[@]}"; do
  echo -e "\n--- Running adaptive with lambda_cost = $LAMBDA ---"
  python main.py --agg_rule=adaptive --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=$ATTACK --lambda_cost=$LAMBDA
done

# --- Run baseline rules for comparison in the same scenario ---
echo -e "\n--- Running baseline rules for comparison ---"
python main.py --agg_rule=fed_avg --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=$ATTACK
python main.py --agg_rule=median --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=$ATTACK
python main.py --agg_rule=krum --num_clients=$CLIENTS --num_malicious=$MALICIOUS --beta=$BETA --num_rounds=$ROUNDS --seed=$SEED --attack_type=$ATTACK

echo -e "\n--- Experiment 3 Complete ---"