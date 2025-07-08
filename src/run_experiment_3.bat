@ECHO OFF

ECHO --- Starting Experiment 3: Tuning FedStrategist's Risk Tolerance (lambda_cost) ---

REM --- Scenario: Moderate Heterogeneity, Stealth Attack ---
set BETA=0.5
set MALICIOUS=5
set CLIENTS=20
set ROUNDS=50
set SEED=2024
set ATTACK=stealth

ECHO.
ECHO --- Running Stealth Attack Scenario: beta=%BETA%, malicious=%MALICIOUS%, attack=%ATTACK% ---

REM --- Run FedStrategist with different lambda values ---
FOR %%L IN (0.1 0.5 1.0 2.0) DO (
  ECHO.
  ECHO --- Running adaptive with lambda_cost = %%L ---
  python main.py --agg_rule=adaptive --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=%ATTACK% --lambda_cost=%%L
)

REM --- Run baseline rules for comparison in the same scenario ---
ECHO.
ECHO --- Running baseline rules for comparison ---
python main.py --agg_rule=fed_avg --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=%ATTACK%
python main.py --agg_rule=median --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=%ATTACK%
python main.py --agg_rule=krum --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=%ATTACK%

ECHO.
ECHO --- Experiment 3 Complete ---