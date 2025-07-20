import numpy as np
import pandas as pd
from load_dataset import load_data
import argparse
import os
from tqdm import tqdm
from methods.statistic import label2res

from statistic import compute_statistics

def importance_sampling(
    df: pd.DataFrame,
    confidence_col: str,
    gold_label_col: str,
    gpt_label_col: str,
    sample_sizes: list,
    repeat: int = 1000,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Perform importance sampling to estimate dataset-specific statistics using model confidence as sampling weights.
    """
    p = df["proxy_label"] / df["proxy_label"].sum() 
    
    if dataset_name != "mt-bench":
        df["gt_label_res"] = [label2res(label, dataset_name) for label in df[gold_label_col]]
        df["gpt_label_res"] = [label2res(label, dataset_name) for label in df[gpt_label_col]]
    else:
        df["gt_label_res"] = [label2res([row["model_a"], row["model_b"], row[gold_label_col]], dataset_name) for _, row in df.iterrows()]
        df["gpt_label_res"] = [label2res([row["model_a"], row["model_b"], row[gpt_label_col]], dataset_name) for _, row in df.iterrows()]
        
    true_statistic = np.mean(df["gt_label_res"])

    results = {"Human_Samples": [], "AvgRelativeError": []}

    for n_samples in tqdm(sample_sizes):
        errors = []

        for _ in range(repeat):
            sampled_indices = np.random.choice(len(df), size=n_samples, replace=True, p=p)
            sampled_df = df.loc[sampled_indices]
            sampled_weights = p[sampled_indices]
            
            estimate = np.mean(sampled_df["gt_label_res"] / sampled_weights / len(df))

            error = abs(estimate - true_statistic) / true_statistic
            errors.append(error)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        results["Human_Samples"].append(n_samples)
        results["AvgRelativeError"].append(rmse)

    return pd.DataFrame(results)

def run_importance_sampling(dataset_name: str, max_human_labels: int, step_size: int, repeat: int, save_dir: str):
    """
    Run importance sampling on a dataset.
    """
    dataset, _ = load_data(dataset_name)
    confidence_col = "confidence_normalized"
    gold_label_col = "gold_label"
    gpt_label_col = "gpt_label"

    total_samples = len(dataset)
    max_size = min(max_human_labels, total_samples)

    sample_sizes = list(range(step_size, max_size + 1, step_size))
    results = importance_sampling(
        df=dataset,
        confidence_col=confidence_col,
        gold_label_col=gold_label_col,
        gpt_label_col=gpt_label_col,
        sample_sizes=sample_sizes,
        repeat=repeat,
        dataset_name=dataset_name
    )

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset_name}_importance_sampling.csv")

    results.insert(0, "Dataset", dataset_name)
    results.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run importance sampling on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (as defined in load_data)")
    parser.add_argument("--max_human_labels", type=int, default=2000, help="Maximum number of human-labeled samples")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for human budgets")
    parser.add_argument("--repeat", type=int, default=1000, help="Number of iterations for stability")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the result")
    args = parser.parse_args()

    run_importance_sampling(
        dataset_name=args.dataset,
        max_human_labels=args.max_human_labels,
        step_size=args.step_size,
        repeat=args.repeat,
        save_dir=args.save_dir
    )