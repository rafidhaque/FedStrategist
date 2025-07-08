# --- File: plot_results.py ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

def plot_accuracy_curves(df, scenario_name):
    """Plots accuracy over rounds for all aggregation rules in a scenario."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='round', y='accuracy', hue='agg_rule_base', style='agg_rule_base', markers=True, dashes=False)
    
    plt.title(f'Model Accuracy vs. Communication Rounds\n({scenario_name})', fontsize=16)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Model Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend(title='Aggregation Rule')
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    save_path = os.path.join(PLOTS_DIR, f'accuracy_{scenario_name}.png')
    plt.savefig(save_path)
    print(f"Saved accuracy plot to {save_path}")
    plt.close()

def main():
    all_files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    if not all_files:
        print("No CSV files found in the 'results' directory.")
        return

    # Create a list of dataframes, adding the source filename to each
    list_of_dfs = []
    for f in all_files:
        df_temp = pd.read_csv(f)
        # Add the filename as a column to parse from later
        df_temp['source_file'] = os.path.basename(f)
        list_of_dfs.append(df_temp)

    df = pd.concat(list_of_dfs, ignore_index=True)

    # Extract scenario and rule from the 'source_file' column.
    # Using the .str accessor is more efficient and idiomatic in pandas than .apply.
    df['agg_rule_base'] = df['source_file'].str.split('rule=').str[1].str.split('_').str[0]

    beta_series = df['source_file'].str.split('beta=').str[1].str.split('_').str[0]
    malicious_series = df['source_file'].str.split('malicious=').str[1].str.split('_').str[0]
    df['scenario'] = 'beta=' + beta_series + '_malicious=' + malicious_series

    # Plot for each scenario
    for scenario, scenario_df in df.groupby('scenario'):
        plot_accuracy_curves(scenario_df, scenario)

if __name__ == '__main__':
    main()