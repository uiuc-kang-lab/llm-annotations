import sys
import numpy as np
from load_dataset import load_data
from statistic import compute_statistics
import argparse

def run_llm_human_confidence(dataset: str, human_budget: float, use_confidence: bool=False, repeat: int=100):
    data, groundtruth = load_data(dataset)
    
    human_label_size = int(human_budget * len(data))
    if use_confidence:
        # find the index of data with top human_budget confidence
        data.sort_values(by="gpt_confidence", ascending=False, inplace=True)
        data["label"] = np.concatenate([data["gold_label"].to_numpy()[:human_label_size], data["gpt_label"].to_numpy()[human_label_size:]])
        try:
            estimate = compute_statistics(data, dataset)
        except:
            estimate = 0
            return None, len(data), human_label_size
        relative_error = abs(estimate - groundtruth) / groundtruth
    else:
        relative_errors = []
        for _ in range(repeat):
            data = data.sample(frac=1)
            data["label"] = np.concatenate([data["gold_label"].to_numpy()[:human_label_size], data["gpt_label"].to_numpy()[human_label_size:]])
            try:
                estimate = compute_statistics(data, dataset)
            except:
                estimate = 0
                continue
            relative_errors.append(abs(estimate - groundtruth) / groundtruth)
        relative_error = np.sqrt(np.mean(np.array(relative_errors)**2))
    return relative_error, len(data), human_label_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLM only or LLM with human in the loop')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--budget', type=str, default="all", help='Human budget')
    parser.add_argument('--use_confidence', type=str, default=False, help='Use confidence or not')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the result')
    args = parser.parse_args()
    args.use_confidence = args.use_confidence == "True"
    if args.budget == "all":
        budgets = [0.001 * i for i in range(1, 10)] + \
            [0.01 * i for i in range(1, 10)] + [0.1 * i for i in range(1, 10)] + [1]
        for budget in budgets:
            relative_error, cost_llm, cost_human = run_llm_human_confidence(args.dataset, human_budget=float(budget), use_confidence=args.use_confidence)
            if args.save_path:
                with open(args.save_path, "a") as f:
                    f.write(f"{args.dataset},{budget},{relative_error},{cost_llm},{cost_human}\n")
            else:
                print(f"Relative error of LLM only inference: {relative_error}")
                print(f"Cost: {cost_llm} llm and {cost_human} human")
                
    else:
        relative_error, cost_llm, cost_human = run_llm_human_confidence(args.dataset, human_budget=float(args.budget), use_confidence=args.use_confidence)
        if args.save_path:
            with open(args.save_path, "a") as f:
                f.write(f"{args.dataset},{args.budget},{relative_error},{cost_llm},{cost_human}\n")
        else:
            print(f"Relative error of LLM only inference: {relative_error}")
            print(f"Cost: {cost_llm} llm and {cost_human} human")