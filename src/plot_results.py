# --- File: plot_results.py ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

def parse_filename(filename):
    """
    Parses experiment parameters from a CSV filename using regex.
    Handles standard, stealth, and lambda-variant filenames.
    """
    pattern = re.compile(
        r"rule=(?P<agg_rule_base>[\w]+)_"
        r"clients=(?P<num_clients>\d+)_"
        r"malicious=(?P<num_malicious>\d+)_"
        r"beta=(?P<beta>[\d\.]+)_"
        r"seed=(?P<seed>\d+)_"
        r"attack=(?P<attack_type>[\w]+)"
        r"(?:_lambda=(?P<lambda_cost>[\d\.]+))?\.csv"  # Optional lambda
    )
    match = pattern.match(filename)
    if not match:
        return None
    
    params = match.groupdict()
    # Fill in default for lambda if not present, and convert to float
    if params['lambda_cost'] is None:
        params['lambda_cost'] = 'N/A'
    else:
        params['lambda_cost'] = float(params['lambda_cost'])
        
    return params

def plot_accuracy_curves(df, scenario_name, scenario_title):
    """Plots accuracy over rounds for all aggregation rules in a scenario."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    sns.lineplot(data=df, x='round', y='accuracy', hue='agg_rule_base', style='agg_rule_base', markers=True, markersize=8, ax=ax)
    
    ax.set_title(f'Model Accuracy vs. Communication Rounds\n({scenario_title})', fontsize=18, pad=20)
    ax.set_xlabel('Communication Round', fontsize=14)
    ax.set_ylabel('Global Model Accuracy (%)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(title='Aggregation Rule', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, f'accuracy_{scenario_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

def plot_lambda_experiment(df, scenario_name, scenario_title):
    """Creates a 2-part plot for the lambda experiment."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Part 1: Accuracy Curves ---
    adaptive_df = df[df['agg_rule_base'] == 'adaptive'].copy()
    adaptive_df['legend'] = 'adaptive (λ=' + adaptive_df['lambda_cost'].astype(str) + ')'
    baselines_df = df[df['agg_rule_base'] != 'adaptive']

    sns.lineplot(data=adaptive_df, x='round', y='accuracy', hue='legend', ax=ax1, marker='o', markersize=6)
    sns.lineplot(data=baselines_df, x='round', y='accuracy', hue='agg_rule_base', ax=ax1, linestyle='--', marker='x')
    
    ax1.set_title(f'Impact of λ_cost on Accuracy under Stealth Attack\n({scenario_title})', fontsize=18, pad=20)
    ax1.set_xlabel('Communication Round', fontsize=14)
    ax1.set_ylabel('Global Model Accuracy (%)', fontsize=14)
    ax1.legend(title='Strategy', fontsize=12, title_fontsize=14)
    
    # --- Part 2: Action Distribution ---
    action_df = df[df['agg_rule_base'] == 'adaptive'][['lambda_cost', 'agg_rule']]
    action_counts = action_df.groupby('lambda_cost')['agg_rule'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    
    sns.barplot(data=action_counts, x='lambda_cost', y='percentage', hue='agg_rule', ax=ax2, palette='viridis')
    
    ax2.set_title('Agent Strategy vs. λ_cost (Risk Tolerance)', fontsize=18, pad=20)
    ax2.set_xlabel('λ_cost (Higher value = More risk-averse)', fontsize=14)
    ax2.set_ylabel('Percentage of Rounds Chosen (%)', fontsize=14)
    ax2.legend(title='Chosen Rule', fontsize=12, title_fontsize=14)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=10)

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(PLOTS_DIR, f'lambda_experiment_{scenario_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved lambda experiment plot to {save_path}")
    plt.close(fig)

def main():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    all_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    if not all_files:
        print(f"Error: No CSV files found in the '{RESULTS_DIR}' directory.")
        return

    all_data = []
    for filename in all_files:
        params = parse_filename(filename)
        if params:
            try:
                df_temp = pd.read_csv(os.path.join(RESULTS_DIR, filename))
                for key, value in params.items():
                    df_temp[key] = value
                all_data.append(df_temp)
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty file {filename}")
            
    if not all_data:
        print("No files matched the expected naming convention.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Group by all scenario parameters to create a plot for each unique experiment setup
    grouping_cols = ['num_clients', 'num_malicious', 'beta', 'attack_type', 'seed']
    
    for scenario_params, scenario_df in full_df.groupby(grouping_cols):
        scenario_name = '_'.join([f"{k}={v}" for k,v in zip(grouping_cols, scenario_params)])
        scenario_title = ', '.join([f"{k.replace('_', ' ').title()}={v}" for k,v in zip(grouping_cols, scenario_params)])

        # Check if this is a lambda experiment
        adaptive_runs = scenario_df[scenario_df['agg_rule_base'] == 'adaptive']
        unique_lambdas = adaptive_runs['lambda_cost'].nunique()

        if unique_lambdas > 1:
            print(f"\nGenerating lambda experiment plot for: {scenario_title}")
            plot_lambda_experiment(scenario_df, scenario_name, scenario_title)
        else:
            print(f"\nGenerating standard accuracy plot for: {scenario_title}")
            plot_accuracy_curves(scenario_df, scenario_name, scenario_title)

if __name__ == '__main__':
    main()