import numpy as np
import pandas as pd

def compute_importance_weights(df, confidence_scores, a=0.5):
    N = len(df)
    uniform_weights = np.ones(N) / N
    confidence_weights = confidence_scores / confidence_scores.sum()
    blended_weights = a * uniform_weights + (1 - a) * confidence_weights
    df['importance'] = blended_weights
    return df

def importance_sampling_control_variate(df, n_samples=100):
    sampled_indices = np.random.choice(df.index, size=n_samples, p=df['importance'], replace=True)
    sampled_data = df.loc[sampled_indices]
    m_hat = np.mean(sampled_data['target'])
    t_hat = np.mean(sampled_data['proxy'])
    t = np.mean(df['proxy'])
    cov = np.cov(sampled_data['proxy'], sampled_data['target'])[0, 1]
    var_t = np.var(sampled_data['proxy'])
    c_hat = - cov / var_t
    final_estimate = len(df) * (m_hat + c_hat * (t_hat - t))
    return final_estimate