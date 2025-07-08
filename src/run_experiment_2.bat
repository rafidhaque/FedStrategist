@echo off
echo "--- Starting Experiment 2: Robustness to Stealth Attack ---"

REM --- Scenario: Moderate Heterogeneity, Stealth Attack ---
set BETA=0.5
set MALICIOUS=5
set CLIENTS=20
set ROUNDS=50
set SEED=123

echo.
echo "--- Running Stealth Attack Scenario: beta=%BETA%, malicious=%MALICIOUS% ---"

python main.py --agg_rule=adaptive --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth
python main.py --agg_rule=fed_avg --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth
python main.py --agg_rule=median --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth
python main.py --agg_rule=krum --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth

echo.
echo "--- Experiment 2 Complete ---"