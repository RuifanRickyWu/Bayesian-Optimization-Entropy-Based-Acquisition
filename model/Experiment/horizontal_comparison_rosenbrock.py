import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def process_folder(folder_path, column_name='best_observation'):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)
    observation_lists = [df[column_name].reset_index(drop=True) for df in all_data]
    observations_df = pd.concat(observation_lists, axis=1)
    observations_df.columns = [f'run_{i + 1}' for i in range(len(observation_lists))]
    mean_series = observations_df.mean(axis=1)
    std_series = observations_df.std(axis=1)
    n = observations_df.shape[1]
    stderr = std_series / np.sqrt(n)
    t_score = stats.t.ppf((1 + 0.95) / 2, df=n - 1)
    margin_of_error = t_score * stderr
    ci_upper = mean_series + margin_of_error
    ci_lower = mean_series - margin_of_error
    result_df = pd.DataFrame({
        'iteration': range(1, len(mean_series) + 1),
        'mean_observation': mean_series,
        'std_observation': std_series,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    return result_df

dimensions = ['5d', '10d', '20d']
methods = ['ES', 'LCB', 'EI', 'PI', 'MES', 'PES']
base_path = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results-rosenbrock'
colors = {
    'ES': 'blue',
    'LCB': 'orange',
    'EI': 'green',
    'PI': 'red',
    'MES': 'gold',
    'PES': 'purple'
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

for idx, dim in enumerate(dimensions):
    ax = axes[idx]
    for method in methods:
        folder_path = os.path.join(f'{base_path}-{dim}', method)
        if not os.path.exists(folder_path):
            print(f'Warning: {folder_path} not found.')
            continue
        result = process_folder(folder_path)
        color = colors.get(method, None)
        ax.semilogy(result['iteration'], result['mean_observation'], label=method, color=color)
        ax.fill_between(result['iteration'], result['ci_lower'], result['ci_upper'], alpha=0.15, color=color)

    ax.set_title(f'Rosenbrock {dim}')
    ax.set_xlabel('Iteration')
    if idx == 0:
        ax.set_ylabel('Best Observation')
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('rosenbrock_comparison_independent_yaxis.svg')
plt.show()