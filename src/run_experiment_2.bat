@echo off
REM This command prevents the commands themselves from being displayed in the console.

echo "--- Starting Trial Test 1: Checking Setup ---"

REM --- Scenario: Moderate Heterogeneity, Stealth Attack (Reduced Workload) ---
set BETA=0.5
set MALICIOUS=5
set CLIENTS=20
REM Reduced from 50 to 5 for a quick test
set ROUNDS=5 
set SEED=123

REM Print a blank line for better readability
echo.
echo "--- Running Trial Scenario: beta=%BETA%, malicious=%MALICIOUS%, rounds=%ROUNDS% ---"

REM --- Running only the 'adaptive' rule for a quick check ---
python main.py --agg_rule=adaptive --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth

REM --- The following rules are commented out for the trial run ---
REM python main.py --agg_rule=fed_avg --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth
REM python main.py --agg_rule=median --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth
REM python main.py --agg_rule=krum --num_clients=%CLIENTS% --num_malicious=%MALICIOUS% --beta=%BETA% --num_rounds=%ROUNDS% --seed=%SEED% --attack_type=stealth

REM Print a blank line before the final message
echo.
echo "--- Trial Test 1 Complete ---"
