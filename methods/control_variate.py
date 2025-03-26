import numpy as np
import pandas as pd
from load_dataset import load_data
import argparse


def control_variate_sampling(
    df: pd.DataFrame,
    confidence_col: str,
    gold_label_col: str,
    gpt_label_col: str,
    sample_sizes: list,
    repeat: int = 1000
) -> pd.DataFrame:
    """
    Perform control variate sampling using model confidence as a proxy variable.

    Parameters:
    - df: The full dataset with predictions, gold labels, and confidence scores.
    - confidence_col: Column name for GPT confidence scores (proxy).
    - gold_label_col: Column name for ground-truth human labels.
    - gpt_label_col: Column name for GPT predictions.
    - sample_sizes: List of sample sizes (human budgets).
    - repeat: Number of iterations for stability.

    Returns:
    - DataFrame of relative errors per sample size.
    """

    # Ground-truth statistic: full accuracy
    true_accuracy = (df[gold_label_col] == df[gpt_label_col]).mean()
    t_full = df[confidence_col].mean()

    results = {"Human Samples": [], "Relative Error": []}

    for n_samples in sample_sizes:
        errors = []

        for _ in range(repeat):
            # Uniformly sample n_samples rows
            sampled = df.sample(n=n_samples, replace=True).copy()

            # Correctness (1 if correct, 0 if incorrect)
            correctness = (sampled[gold_label_col] == sampled[gpt_label_col]).astype(float)
            t_hat = sampled[confidence_col].mean()
            m_hat = correctness.mean()

            # Control variate coefficient
            cov = np.cov(sampled[confidence_col], correctness, ddof=0)[0, 1]
            var_t = np.var(sampled[confidence_col], ddof=0)
            c_hat = -cov / var_t if var_t > 1e-6 else 0  # regularized

            # Adjusted estimate
            adjusted = m_hat + c_hat * (t_hat - t_full)

            # Relative error
            error = abs(adjusted - true_accuracy) / true_accuracy
            errors.append(error)

        results["Human Samples"].append(n_samples)
        results["Relative Error"].append(np.sqrt(np.mean(np.array(errors) ** 2)))  # RMSE

    return pd.DataFrame(results)


def run_control_variate(dataset_name, step_size, max_human_budget, repeat):
    """
    Run control variate sampling on a dataset.

    Args:
        dataset_name: Name of the dataset (as defined in load_data).
        step_size: Step size for human budgets.
        max_human_budget: Maximum number of samples to label with human data.
        repeat: Number of repetitions for stability.

    Returns:
        None
    """
    # Load the dataset
    df, _ = load_data(dataset_name)

    # Define column names
    confidence_col = "confidence_normalized"
    gold_label_col = "gold_label"
    gpt_label_col = "gpt_label"

    # Generate sample sizes
    sample_sizes = list(range(step_size, max_human_budget + 1, step_size))

    # Run control variate sampling
    results = control_variate_sampling(
        df=df,
        confidence_col=confidence_col,
        gold_label_col=gold_label_col,
        gpt_label_col=gpt_label_col,
        sample_sizes=sample_sizes,
        repeat=repeat
    )

    # Print results
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control variate sampling on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (as defined in load_data)")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for human budgets")
    parser.add_argument("--max_human_budget", type=int, default=1000, help="Maximum number of human-labeled samples")
    parser.add_argument("--repeat", type=int, default=50, help="Number of repetitions for stability")
    args = parser.parse_args()

    run_control_variate(
        dataset_name=args.dataset,
        step_size=args.step_size,
        max_human_budget=args.max_human_budget,
        repeat=args.repeat
    )