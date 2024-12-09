import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_target_statistic(df, target_col):
    total_target_statistic = df[target_col].sum()
    return total_target_statistic


def compute_importance_weights(df, confidence_col, a=0.5):
    N = len(df)
    uniform_weights = np.ones(N) / N 
    confidence_scores = df[confidence_col].values
    normalized_confidence = confidence_scores / confidence_scores.sum() 
    blended_weights = a * uniform_weights + (1 - a) * normalized_confidence
    df['importance'] = blended_weights
    return df


def importance_sampling(df, target_col, n_samples):
    sampled_indices = np.random.choice(df.index, size=n_samples, p=df['importance'].values, replace=True)
    sampled_data = df.loc[sampled_indices]
    weights = 1 / (sampled_data['importance'] * len(df))  
    unweighted_estimate = np.sum(sampled_data[target_col] * weights) 
    return unweighted_estimate


def calculate_relative_error(estimated, true_value):
    return abs(estimated - true_value) / true_value


def run_importance_sampling_analysis(df, target_col, confidence_col, sample_sizes, a=0.5, num_iterations=50):
    true_target_statistic = compute_target_statistic(df, target_col)

    df = compute_importance_weights(df, confidence_col, a)

    results = {'sample_size': [], 'relative_error': []}
    for n_samples in sample_sizes:
        estimates = [importance_sampling(df, target_col, n_samples) for _ in range(num_iterations)]
        mean_estimate = np.mean(estimates)
        relative_error = calculate_relative_error(mean_estimate, true_target_statistic)
        results['sample_size'].append(n_samples)
        results['relative_error'].append(relative_error)

    return pd.DataFrame(results)


def plot_relative_error(results_df, title="Relative Error of Importance Sampling"):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['sample_size'], results_df['relative_error'], marker='o', label='Relative Error')
    plt.xlabel('Sample Size (# human labels)')
    plt.ylabel('Relative Error')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()