import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

def compute_win_rate(df, model_name, winner_col):
    model_rows = df[(df["model_a"] == model_name) | (df["model_b"] == model_name)]
    wins = ((model_rows["model_a"] == model_name) & (model_rows[winner_col] == "model_a")) | \
           ((model_rows["model_b"] == model_name) & (model_rows[winner_col] == "model_b"))
    return wins.sum() / len(model_rows)

def convert_to_binary_indicators(df, model_name):
    df = df.copy()
    model_rows = (df["model_a"] == model_name) | (df["model_b"] == model_name)
    df = df[model_rows]
    
    binary_human = (((df["model_a"] == model_name) & (df["gold_label"] == "model_a")) |
                    ((df["model_b"] == model_name) & (df["gold_label"] == "model_b"))).astype(float)
    
    binary_gpt4 = (((df["model_a"] == model_name) & (df["gpt_label"] == "model_a")) |
                   ((df["model_b"] == model_name) & (df["gpt_label"] == "model_b"))).astype(float)
    
    df["binary_human"] = binary_human
    df["binary_gpt4"] = binary_gpt4
    return df

def adjust_with_control_variates(df, sample_indices, tau):
    sampled_df = df.iloc[sample_indices]
    covariance_m_t = np.cov(sampled_df["binary_human"], sampled_df["binary_gpt4"])[0, 1]
    var_t = sampled_df["binary_gpt4"].var()
    c_star = -covariance_m_t / var_t if var_t != 0 else 0
    adjusted_estimate = sampled_df["binary_human"].mean() + c_star * (sampled_df["binary_gpt4"].mean() - tau)
    return adjusted_estimate

def importance_sampling(df, model_name, sample_size, confidence_col="confidence_normalized"):
    model_df = convert_to_binary_indicators(df, model_name)
    weights = model_df[confidence_col] / model_df[confidence_col].sum()
    
    sample_indices = np.random.choice(
        len(model_df), 
        size=min(sample_size, len(model_df)), 
        replace=True, 
        p=weights
    )
    sampled_df = model_df.iloc[sample_indices]
    sampling_probs = weights.iloc[sample_indices].values
    importance_weights = 1.0 / (sampling_probs * len(model_df))
    
    weighted_estimate = (sampled_df["binary_human"] * importance_weights).sum() / importance_weights.sum()
    return weighted_estimate

def run_sampling_analysis(df, model_name, sampling_rates, num_trials=1000):
    win_rate_human = compute_win_rate(df, model_name, "gold_label")
    win_rate_llm = compute_win_rate(df, model_name, "gpt_label")
    relative_error_llm = abs(win_rate_llm - win_rate_human) / win_rate_human * 100

    df = convert_to_binary_indicators(df, model_name)
    np.random.seed(42)

    avg_relative_errors = []
    avg_relative_errors_control_variates = []
    avg_relative_errors_importance = []
    
    for rate in sampling_rates:
        sample_size = int(rate * len(df))
        relative_errors = []
        relative_errors_control_variates = []
        relative_errors_importance = []
        
        for _ in range(num_trials):
            sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
            sampled_df = df.iloc[sample_indices]
            
            # Uniform sampling
            uniform_win_rate = sampled_df["binary_human"].mean()
            relative_error = abs(uniform_win_rate - win_rate_human) / win_rate_human * 100
            relative_errors.append(relative_error)
            
            # Control variates
            adjusted_win_rate = adjust_with_control_variates(df, sample_indices, win_rate_llm)
            relative_error_cv = abs(adjusted_win_rate - win_rate_human) / win_rate_human * 100
            relative_errors_control_variates.append(relative_error_cv)

            # Importance sampling
            importance_win_rate = importance_sampling(df, model_name, sample_size)
            relative_error_imp = abs(importance_win_rate - win_rate_human) / win_rate_human * 100
            relative_errors_importance.append(relative_error_imp)
        
        avg_relative_errors.append(np.mean(relative_errors))
        avg_relative_errors_control_variates.append(np.mean(relative_errors_control_variates))
        avg_relative_errors_importance.append(np.mean(relative_errors_importance))
    
    return {
        'win_rate_human': win_rate_human,
        'win_rate_llm': win_rate_llm,
        'relative_error_llm': relative_error_llm,
        'avg_relative_errors': avg_relative_errors,
        'avg_relative_errors_cv': avg_relative_errors_control_variates,
        'avg_relative_errors_importance': avg_relative_errors_importance
    }

def plot_and_save_results(results, sampling_rates, dataset_size, model_name):
    num_samples = sampling_rates * dataset_size

    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, results['avg_relative_errors'], '-', color='blue', label='Uniform Sampling')
    plt.plot(num_samples, results['avg_relative_errors_cv'], '-', color='green', label='With Control Variates')
    plt.plot(num_samples, results['avg_relative_errors_importance'], '-', color='purple', label='Importance Sampling')
    plt.axhline(y=results['relative_error_llm'], color='red', linestyle='--', label='LLM Baseline')
    
    plt.xlabel("# Human Labels")
    plt.ylabel("Relative Error (%)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Relative Error of Win Rate Estimation for {model_name}")
    plt.savefig(os.path.join("../results/plots/mt-bench", f"{model_name}.png"))
    plt.close()

def save_results_to_csv(results, sampling_rates, dataset_size, model_name):
    num_samples = (sampling_rates * dataset_size).astype(int)

    # Uniform Sampling
    df_uniform = pd.DataFrame({
        "model_name": [model_name] * len(num_samples),
        "num_samples": num_samples,
        "relative_error": results['avg_relative_errors']
    })
    df_uniform.to_csv(os.path.join("../results/uniform_sampling/mt-bench", f"{model_name}_uniform_sampling.csv"), index=False)

    # Control Variates
    df_cv = pd.DataFrame({
        "model_name": [model_name] * len(num_samples),
        "num_samples": num_samples,
        "relative_error": results['avg_relative_errors_cv']
    })
    df_cv.to_csv(os.path.join("../results/control_variate/mt-bench", f"{model_name}_control_variates.csv"), index=False)

    # Importance Sampling
    df_importance = pd.DataFrame({
        "model_name": [model_name] * len(num_samples),
        "num_samples": num_samples,
        "relative_error": results['avg_relative_errors_importance']
    })
    df_importance.to_csv(os.path.join("../results/importance_sampling/mt-bench", f"{model_name}_importance_sampling.csv"), index=False)

def analyze_model(model_name, df, num_trials=1000):
    sampling_rates = np.linspace(0.001, 0.2, 20)[1:]
    results = run_sampling_analysis(df, model_name, sampling_rates, num_trials)

    print(f"\nResults for {model_name}:")
    print(f"Win Rate (Human Judges): {results['win_rate_human']:.3f}")
    print(f"Win Rate (LLM Judges): {results['win_rate_llm']:.3f}")
    print(f"Relative Error (LLM vs. Human): {results['relative_error_llm']:.2f}%")

    plot_and_save_results(results, sampling_rates, len(df), model_name)
    save_results_to_csv(results, sampling_rates, len(df), model_name)
    
    return results

def main():
    # Suppress common warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Load data
    judge_df = pd.read_csv("../datasets/mt-bench/mt-bench.csv")

    # List of models
    models = [
        "gpt-3.5-turbo",
        "claude-v1",
        "vicuna-13b-v1.2",
        "llama-13b",
        "gpt-4",
        "alpaca-13b"
    ]

    for model_name in models:
        analyze_model(model_name, judge_df)

if __name__ == "__main__":
    main()
