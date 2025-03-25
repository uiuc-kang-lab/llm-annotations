import numpy as np
import pandas as pd
from load_dataset import load_data
import argparse


def importance_sampling(
    df: pd.DataFrame,
    confidence_col: str,
    gold_label_col: str,
    gpt_label_col: str,
    sample_sizes: list,
    repeat: int = 1000
) -> pd.DataFrame:
    """
    Perform importance sampling to estimate accuracy using model confidence as sampling weights.

    Parameters:
    - df: Dataset with predictions and gold labels
    - confidence_col: Column with model confidence (used as importance weights)
    - gold_label_col: Column with gold labels (human labels)
    - gpt_label_col: Column with LLM predictions
    - sample_sizes: List of sample sizes (human budget)
    - repeat: Number of sampling iterations for stability

    Returns:
    - DataFrame with relative error for each sample size
    """

    # Step 1: Normalize confidence scores
    df["importance"] = df[confidence_col] / df[confidence_col].sum()

    # Step 2: Compute ground-truth accuracy
    true_accuracy = (df[gold_label_col] == df[gpt_label_col]).sum() / len(df)

    results = {"Human Samples": [], "Relative Error": []}

    for n_samples in sample_sizes:
        errors = []

        for _ in range(repeat):
            # Step 3: Sample using importance weights
            sampled_indices = np.random.choice(df.index, size=n_samples, p=df["importance"].values, replace=True)
            sampled = df.loc[sampled_indices]

            # Step 4: Accuracy from gold vs. gpt labels in the sample
            correct = (sampled[gold_label_col] == sampled[gpt_label_col]).sum()
            estimate = correct / n_samples

            # Step 5: Relative error
            error = abs(estimate - true_accuracy) / true_accuracy
            errors.append(error)

        results["Human Samples"].append(n_samples)
        results["Relative Error"].append(np.sqrt(np.mean(np.array(errors) ** 2)))  # RMSE

    return pd.DataFrame(results)


def run_importance_sampling(dataset_name: str, max_human_budget: int, step_size: int, repeat: int):
    """
    Run importance sampling on a dataset.

    Args:
        dataset_name: Name of the dataset (as defined in load_data).
        max_human_budget: Maximum number of samples to label with human (gold) labels.
        step_size: Step size for human budgets.
        repeat: Number of sampling repetitions.

    Returns:
        None
    """
    # Load the dataset and compute the true statistic
    dataset, _ = load_data(dataset_name)

    # Define column names (adjust these if your dataset uses different names)
    confidence_col = "confidence_normalized"
    gold_label_col = "gold_label"
    gpt_label_col = "gpt_label"

    # Generate sample sizes in steps of `step_size`
    sample_sizes = list(range(step_size, max_human_budget + 1, step_size))

    # Run importance sampling
    results = importance_sampling(
        df=dataset,
        confidence_col=confidence_col,
        gold_label_col=gold_label_col,
        gpt_label_col=gpt_label_col,
        sample_sizes=sample_sizes,
        repeat=repeat
    )

    # Print results
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run importance sampling on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (as defined in load_data)")
    parser.add_argument("--max_human_budget", type=int, default=1000, help="Maximum number of human-labeled samples")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for human budgets")
    parser.add_argument("--repeat", type=int, default=1000, help="Number of iterations for stability")
    args = parser.parse_args()

    run_importance_sampling(
        dataset_name=args.dataset,
        max_human_budget=args.max_human_budget,
        step_size=args.step_size,
        repeat=args.repeat
    )