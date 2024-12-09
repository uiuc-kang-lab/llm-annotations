import numpy as np
import pandas as pd

def control_variate_testing(df, target_col, proxy_col, n_samples=100):
    sampled_indices = np.random.choice(df.index, size=n_samples, replace=True)
    sampled_data = df.loc[sampled_indices]

    m_hat = np.mean(sampled_data[target_col])
    t_hat = np.mean(sampled_data[proxy_col])
    t = np.mean(df[proxy_col])

    cov = np.cov(sampled_data[proxy_col], sampled_data[target_col])[0, 1]
    var_t = np.var(sampled_data[proxy_col])

    c_hat = - cov / var_t if var_t != 0 else 0

    final_estimate = len(df) * (m_hat + c_hat * (t_hat - t))

    return final_estimate

def run_control_variate_analysis(df, target_col, proxy_col, sample_sizes, num_iterations=50):
    results = {'sample_size': [], 'mean_estimate': [], 'relative_error': []}

    true_statistic = df[target_col].sum()

    for n_samples in sample_sizes:
        estimates = [control_variate_testing(df, target_col, proxy_col, n_samples) for _ in range(num_iterations)]
        mean_estimate = np.mean(estimates)

        relative_error = np.abs(mean_estimate - true_statistic) / true_statistic if true_statistic != 0 else 0

        results['sample_size'].append(n_samples)
        results['mean_estimate'].append(mean_estimate)
        results['relative_error'].append(relative_error)

    results_df = pd.DataFrame(results)
    return results_df