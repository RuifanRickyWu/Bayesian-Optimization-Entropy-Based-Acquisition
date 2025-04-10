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
    observations_df.columns = [f'run_{i+1}' for i in range(len(observation_lists))]

    mean_series = observations_df.mean(axis=1)
    std_series = observations_df.std(axis=1)

    # 95% confidence interval
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

folder_path_es = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/ES'
folder_path_lcb = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/LCB'
folder_path_pi = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/PI'
folder_path_ei = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/EI'
folder_path_mes = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/MES'
folder_path_pes = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/PES'
folder_path_random = '/Users/rwu/dev/thesis/thesis_project_information_based_acquisition_functions/results/Random'

es_result = process_folder(folder_path_es, column_name='best_observation')
print(es_result)
lcb_result = process_folder(folder_path_lcb, column_name='best_observation')
ei_result = process_folder(folder_path_ei, column_name='best_observation')
pi_result = process_folder(folder_path_pi, column_name='best_observation')
mes_result = process_folder(folder_path_mes, column_name='best_observation')
pes_result = process_folder(folder_path_pes, column_name='best_observation')
random_result = process_folder(folder_path_random, column_name='best_observation')

plt.figure(figsize=(10, 6))

plt.semilogy(es_result['iteration'], es_result['mean_observation'], label='ES Mean', color='blue')
plt.fill_between(es_result['iteration'], es_result['ci_lower'], es_result['ci_upper'], alpha=0.2, color='blue')#, label='ES 95% CI')

plt.plot(lcb_result['iteration'], lcb_result['mean_observation'], label='LCB Mean', color='orange')
plt.fill_between(lcb_result['iteration'], lcb_result['ci_lower'], lcb_result['ci_upper'], alpha=0.2, color='orange')#, label='LCB 95% CI')

plt.plot(ei_result['iteration'], ei_result['mean_observation'], label='EI Mean', color='green')
plt.fill_between(ei_result['iteration'], ei_result['ci_lower'], ei_result['ci_upper'], alpha=0.2, color='green')#, label='EI 95% CI')

plt.plot(pi_result['iteration'], pi_result['mean_observation'], label='PI Mean', color='red')
plt.fill_between(pi_result['iteration'], pi_result['ci_lower'], pi_result['ci_upper'], alpha=0.2, color='red')#, label='PI 95% CI')

plt.plot(mes_result['iteration'], mes_result['mean_observation'], label='MES Mean', color='yellow')
plt.fill_between(mes_result['iteration'], mes_result['ci_lower'], mes_result['ci_upper'], alpha=0.2, color='yellow')#, label='MES 95% CI')


plt.plot(pes_result['iteration'], pes_result['mean_observation'], label='PES Mean', color='purple')
plt.fill_between(pes_result['iteration'], pes_result['ci_lower'], pes_result['ci_upper'], alpha=0.2, color='purple')#, label='PES 95% CI')

plt.xlabel('Iteration')
plt.ylabel('Best Observation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Mean and 95% Confidence Interval Across 10 Runs.svg')
plt.show()
