import numpy as np
import pandas as pd
from load_dataset import load_data
from statistic import compute_statistics
import argparse
import os

def control_variate_sampling(
    df: pd.DataFrame,
    confidence_col: str,
    gold_label_col: str,
    gpt_label_col: str,
    sample_sizes: list,
    repeat: int = 1000,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Perform control variate sampling using model confidence as a proxy variable.
    """
    true_statistic = compute_statistics(df, dataset_name, label_column=gpt_label_col)
    t_full = df[confidence_col].mean()

    results = {"Human Samples": [], "Relative Error": []}

    for n_samples in sample_sizes:
        errors = []

        for _ in range(repeat):
            sampled = df.sample(n=n_samples, replace=True).copy()
            t_hat = sampled[confidence_col].mean()

            estimate = compute_statistics(sampled, dataset_name, label_column=gpt_label_col)
            correctness = (sampled[gold_label_col] == sampled[gpt_label_col]).astype(float)

            cov = np.cov(sampled[confidence_col], correctness, ddof=0)[0, 1]
            var_t = np.var(sampled[confidence_col], ddof=0)
            c_hat = -cov / var_t if var_t > 1e-6 else 0  # Mitigate any division-by-zero

            adjusted = estimate + c_hat * (t_hat - t_full)
            error = abs(adjusted - true_statistic) / true_statistic if true_statistic else float('inf')
            errors.append(error)

        results["Human Samples"].append(n_samples)
        results["Relative Error"].append(np.sqrt(np.mean(np.array(errors) ** 2)))  # RMSE

    return pd.DataFrame(results)

def run_control_variate(dataset_name, step_size, max_human_budget, repeat, save_dir):
    """
    Run control variate sampling on a dataset and save to CSV.
    """
    df, _ = load_data(dataset_name)
    confidence_col = "confidence_normalized"
    gold_label_col = "gold_label"
    gpt_label_col = "gpt_label"

    total_samples = len(df)
    max_size = min(max_human_budget, total_samples)

    # Build a sequence of sample sizes from step_size to max_size
    sample_sizes = list(range(step_size, max_size + 1, step_size))

    results = control_variate_sampling(
        df=df,
        confidence_col=confidence_col,
        gold_label_col=gold_label_col,
        gpt_label_col=gpt_label_col,
        sample_sizes=sample_sizes,
        repeat=repeat,
        dataset_name=dataset_name
    )

    # Save results to CSV
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset_name}_control_variate.csv")
    results.insert(0, "Dataset", dataset_name)
    results.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control variate sampling on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (as defined in load_data)")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for human budgets")
    parser.add_argument("--max_human_budget", type=int, default=1000, help="Maximum number of human-labeled samples")
    parser.add_argument("--repeat", type=int, default=50, help="Number of repetitions for stability")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the results")
    args = parser.parse_args()

    run_control_variate(
        dataset_name=args.dataset,
        step_size=args.step_size,
        max_human_budget=args.max_human_budget,
        repeat=args.repeat,
        save_dir=args.save_dir
    )