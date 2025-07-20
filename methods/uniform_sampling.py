import sys
import numpy as np
from methods.load_dataset import load_data
from methods.statistic import compute_statistics
import argparse
from tqdm import tqdm
import os

def run_uniform_sampling(dataset: str, step_size: int, repeat: int, save_dir: str, max_human_labels: int = 2000):
    data, groundtruth = load_data(dataset)
    total_size = min(len(data), max_human_labels)

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset}_uniform_sampling.csv")

    with open(output_file, "w") as f:
        f.write("Dataset,Human_Samples,AvgRelativeError\n")

    for human_samples in tqdm(range(step_size, total_size + 1, step_size)):
        relative_errors = []

        for _ in range(repeat):
            sampled_data = data.sample(n=human_samples, random_state=None).reset_index(drop=True)

            try:
                estimate = compute_statistics(sampled_data, dataset, label_column="gold_label")
            except Exception as e:
                print(f"Error computing statistics: {e}")
                estimate = 0

            if groundtruth != 0:
                relative_errors.append(abs(estimate - groundtruth) / groundtruth)
            else:
                relative_errors.append(float('inf'))

        avg_relative_error = np.mean(relative_errors)

        with open(output_file, "a") as f:
            f.write(f"{dataset},{human_samples},{avg_relative_error}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run uniform sampling on a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--step_size', type=int, default=100, help='Number of samples to increment at each step')
    parser.add_argument('--repeat', type=int, default=10, help='Number of iterations for each step size')
    parser.add_argument('--save_dir', type=str, default="./", help='Directory to save the result')
    parser.add_argument('--max_human_labels', type=int, default=2000, help='Maximum number of human labels to consider')
    args = parser.parse_args()

    run_uniform_sampling(args.dataset, step_size=args.step_size, repeat=args.repeat, save_dir=args.save_dir, max_human_labels=args.max_human_labels)