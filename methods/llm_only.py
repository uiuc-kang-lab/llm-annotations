import argparse, os
from methods.load_dataset import load_data
from methods.statistic import compute_statistics

def run_llm_only(dataset: str):
    data, groundtruth = load_data(dataset)
    estimate = compute_statistics(data, dataset)
    relative_error = abs(estimate - groundtruth) / groundtruth
    return relative_error, len(data), 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run uniform sampling on a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--step_size', type=int, default=100, help='Number of samples to increment at each step')
    parser.add_argument('--repeat', type=int, default=10, help='Number of iterations for each step size')
    parser.add_argument('--save_dir', type=str, default="./", help='Directory to save the result')
    parser.add_argument('--max_human_labels', type=int, default=2000, help='Maximum number of human labels to consider')
    args = parser.parse_args()
    relative_error, cost_llm, cost_human = run_llm_only(args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/{args.dataset}_llm_only.csv", "w") as f:
        f.write("Dataset,Human_Samples,AvgRelativeError\n")
        f.write(f"{args.dataset},{cost_human},{relative_error}\n")
    