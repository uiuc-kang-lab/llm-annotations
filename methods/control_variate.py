import numpy as np
import pandas as pd
from load_dataset import load_data
from statistic import compute_statistics
import argparse
import os
from tqdm import tqdm

from methods.statistic import label2res

def control_variate_sampling(
    df: pd.DataFrame,
    gold_label_col: str,
    gpt_label_col: str,
    sample_sizes: list,
    repeat: int = 1000,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Perform control variate sampling using model confidence as a proxy variable.
    """
    true_statistic = compute_statistics(df, dataset_name, label_column=gold_label_col)
    if dataset_name != "mt-bench":
        df["gt_label_res"] = [label2res(label, dataset_name) for label in df[gold_label_col]]
        df["gpt_label_res"] = [label2res(label, dataset_name) for label in df[gpt_label_col]]
    else:
        df["gt_label_res"] = [label2res([row["model_a"], row["model_b"], row[gold_label_col]], dataset_name) for _, row in df.iterrows()]
        df["gpt_label_res"] = [label2res([row["model_a"], row["model_b"], row[gpt_label_col]], dataset_name) for _, row in df.iterrows()]
        
    t_full = df["gpt_label_res"].mean()  # Full sample mean of the GPT labels

    results = {"Human_Samples": [], "AvgRelativeError": []}

    for n_samples in tqdm(sample_sizes):
        errors = []

        for _ in range(repeat):
            sampled_df = df.sample(n=n_samples, replace=False)
            t_hat = sampled_df["gpt_label_res"].mean()

            estimate = compute_statistics(sampled_df, dataset_name, label_column=gold_label_col)
            
            cov = np.cov(sampled_df["gpt_label_res"], sampled_df["gt_label_res"])[0, 1]
            var_t = np.var(sampled_df["gpt_label_res"], ddof=0)
            c_hat = -cov / var_t

            adjusted = estimate + c_hat * (t_hat - t_full)
            error = abs(adjusted - true_statistic) / true_statistic
            errors.append(error)

        results["Human_Samples"].append(n_samples)
        results["AvgRelativeError"].append(np.mean(errors))  # RMSE

    return pd.DataFrame(results)

def run_control_variate(dataset_name, step_size, max_human_labels, repeat, save_dir):
    """
    Run control variate sampling on a dataset and save to CSV.
    """
    df, _ = load_data(dataset_name)
    confidence_col = "confidence_normalized"
    gold_label_col = "gold_label"
    gpt_label_col = "gpt_label"

    total_samples = len(df)
    max_size = min(max_human_labels, total_samples)

    sample_sizes = list(range(step_size, max_size + 1, step_size))

    results = control_variate_sampling(
        df=df,
        gold_label_col=gold_label_col,
        gpt_label_col=gpt_label_col,
        sample_sizes=sample_sizes,
        repeat=repeat,
        dataset_name=dataset_name
    )

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset_name}_control_variate.csv")
    results.insert(0, "Dataset", dataset_name)
    results.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control variate sampling on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (as defined in load_data)")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for human budgets")
    parser.add_argument("--max_human_labels", type=int, default=2000, help="Maximum number of human-labeled samples")
    parser.add_argument("--repeat", type=int, default=50, help="Number of repetitions for stability")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the results")
    args = parser.parse_args()

    run_control_variate(
        dataset_name=args.dataset,
        step_size=args.step_size,
        max_human_labels=args.max_human_labels,
        repeat=args.repeat,
        save_dir=args.save_dir
    )