import sys
import numpy as np
from methods.load_dataset import load_data
from methods.statistic import compute_statistics
import argparse
import os

def run_uniform_sampling(dataset: str, step_size: int, repeat: int, save_dir: str):
    data, groundtruth = load_data(dataset)
    total_size = len(data)

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset}_uniform_sampling.csv")

    with open(output_file, "w") as f:
        f.write("Dataset,Human_Samples,AvgRelativeError,LLM_Samples\n")

    for human_samples in range(0, total_size + 1, step_size):
        llm_samples = total_size - human_samples
        relative_errors = []

        for _ in range(repeat):
            sampled_data = data.sample(frac=1, random_state=None).reset_index(drop=True)

            if dataset in ["helmet", "global_warming", "mrpc", "med-safe"]:
                sampled_data["label"] = sampled_data["gpt_label"].astype(int, errors="ignore")
            else:
                sampled_data["label"] = sampled_data["gpt_label"].astype(str)

            if human_samples > 0:
                if dataset in ["helmet", "global_warming", "mrpc", "med-safe"]:
                    sampled_data.loc[:human_samples - 1, "label"] = (
                        sampled_data["gold_label"].astype(int, errors="ignore").to_numpy()[:human_samples]
                    )
                else:
                    sampled_data.loc[:human_samples - 1, "label"] = (
                        sampled_data["gold_label"].astype(str).to_numpy()[:human_samples]
                    )

            try:
                estimate = compute_statistics(sampled_data, dataset, label_column="label")
            except Exception as e:
                print(f"Error computing statistics: {e}")
                estimate = 0

            if groundtruth != 0:
                relative_errors.append(abs(estimate - groundtruth) / groundtruth)
            else:
                relative_errors.append(float('inf'))

        avg_relative_error = np.mean(relative_errors)

        with open(output_file, "a") as f:
            f.write(f"{dataset},{human_samples},{avg_relative_error},{llm_samples}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run uniform sampling on a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--step_size', type=int, default=100, help='Number of samples to increment at each step')
    parser.add_argument('--repeat', type=int, default=10, help='Number of iterations for each step size')
    parser.add_argument('--save_dir', type=str, default="./", help='Directory to save the result')
    args = parser.parse_args()

    run_uniform_sampling(args.dataset, step_size=args.step_size, repeat=args.repeat, save_dir=args.save_dir)