@echo off
REM This script is the Windows Batch equivalent of the provided Bash script.

echo "--- Starting Experiment 4: The Krum Crucible ---"

:: --- Scenario: Low Heterogeneity (IID-like), Standard Loud Attack ---
:: We set beta to a high value to make benign clients similar.
:: We use the 'standard' attack to make malicious clients loud outliers.
set BETA=10.0
set MALICIOUS=5
set CLIENTS=20
set ROUNDS=50
set SEED=404
:: A new seed for our final experiment

echo.
echo "--- Running Krum-Favorable Scenario: beta=%BETA%, attack=standard ---"

python main.py --agg_rule=adaptive --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=standard
python main.py --agg_rule=fed_avg --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=standard
python main.py --agg_rule=median --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=standard
python main.py --agg_rule=krum --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=standard

echo.
echo "--- Experiment 4 Complete ---"
