@ECHO OFF

ECHO --- Starting Experiment 1: Benchmarking FedStrategist ---

REM --- Scenario 1: Moderate Heterogeneity, Moderate Attack ---
set BETA=0.5
set MALICIOUS=5
set CLIENTS=20
set ROUNDS=50
set SEED=42

ECHO.
ECHO --- Running Scenario 1: beta=%BETA%, malicious=%MALICIOUS% ---

python main.py --agg_rule=adaptive --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%
python main.py --agg_rule=fed_avg --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%
python main.py --agg_rule=median --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%
python main.py --agg_rule=krum --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%

REM --- Scenario 2: High Heterogeneity (Non-IID), Moderate Attack ---
set BETA=0.1
set MALICIOUS=5

ECHO.
ECHO --- Running Scenario 2: beta=%BETA%, malicious=%MALICIOUS% ---

python main.py --agg_rule=adaptive --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%
python main.py --agg_rule=fed_avg --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%
python main.py --agg_rule=median --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%
python main.py --agg_rule=krum --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED%

ECHO.
ECHO --- Experiment 1 Complete ---