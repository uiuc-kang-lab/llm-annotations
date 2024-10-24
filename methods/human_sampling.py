import sys
import numpy as np
from load_dataset import load_data
from statistic import compute_statistics
import argparse

def run_human_sampling(dataset: str, human_budget: float, repeat: int=100):
    data, groundtruth = load_data(dataset)
    relative_errors = []
    for _ in range(repeat):
        sampled_data = data.sample(frac=human_budget, replace=False)
        sampled_data["label"] = sampled_data["gold_label"]
        try:
            estimate = compute_statistics(sampled_data, dataset)
            relative_error = abs(estimate - groundtruth) / groundtruth
        except Exception as e:
            continue
            
        relative_errors.append(relative_error)
    
    if len(relative_errors) == 0:
        return None, len(data), len(sampled_data)
    relative_error = np.sqrt(np.mean(np.array(relative_errors)**2))
    
    return relative_error, len(data), len(sampled_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run human sampling')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--budget', type=str, default="all", help='Human budget')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the result')
    args = parser.parse_args()
    
    if args.budget == "all":
        budgets = [0.001 * i for i in range(1, 10)] + \
            [0.01 * i for i in range(1, 10)] + [0.1 * i for i in range(1, 10)] + [1]
        for budget in budgets:
            relative_error, cost_llm, cost_human = run_human_sampling(args.dataset, human_budget=float(budget))
            if args.save_path:
                with open(args.save_path, "a") as f:
                    f.write(f"{args.dataset},{budget},{relative_error},{cost_llm},{cost_human}\n")
            else:
                print(f"Relative error of LLM only inference: {relative_error}")
                print(f"Cost: {cost_llm} llm and {cost_human} human")
                
    else:
        relative_error, cost_llm, cost_human = run_human_sampling(args.dataset, human_budget=float(args.budget))
        if args.save_path:
            with open(args.save_path, "a") as f:
                f.write(f"{args.dataset},{args.budget},{relative_error},{cost_llm},{cost_human}\n")
        else:
            print(f"Relative error of LLM only inference: {relative_error}")
            print(f"Cost: {cost_llm} llm and {cost_human} human")