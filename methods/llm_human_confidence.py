import sys
import numpy as np
from load_dataset import load_data
from statistic import compute_statistics
import argparse

def run_llm_human_confidence(dataset: str, human_budget: float, use_confidence: bool=False, repeat: int=100):
    data, groundtruth = load_data(dataset)
    
    human_label_size = int(human_budget * len(data))
    if use_confidence:
        # Confidence-based sampling
        data.sort_values(by="confidence_normalized", ascending=False, inplace=True)
        data["label"] = data["gold_label"].to_numpy()
        data.loc[human_label_size:, "label"] = data["gpt_label"].to_numpy()[human_label_size:]
        try:
            estimate = compute_statistics(data, dataset, label_column="label")
        except Exception as e:
            print(f"Error computing statistics: {e}")
            estimate = 0
            return None, len(data), human_label_size
        relative_error = abs(estimate - groundtruth) / groundtruth
    else:
        # Random sampling
        relative_errors = []
        for _ in range(repeat):
            data = data.sample(frac=1).reset_index(drop=True)
            data["label"] = data["gold_label"].to_numpy()
            data.loc[human_label_size:, "label"] = data["gpt_label"].to_numpy()[human_label_size:]
            try:
                estimate = compute_statistics(data, dataset, label_column="label")
            except Exception as e:
                print(f"Error computing statistics: {e}")
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
                    f.write(f"{args.dataset},{int(cost_human)},{relative_error},{cost_llm},{cost_human}\n")
            else:
                print(f"Relative error of LLM only inference: {relative_error}")
                print(f"Cost: {cost_llm} llm and {int(cost_human)} human samples")
    else:
        relative_error, cost_llm, cost_human = run_llm_human_confidence(args.dataset, human_budget=float(args.budget), use_confidence=args.use_confidence)
        if args.save_path:
            with open(args.save_path, "a") as f:
                f.write(f"{args.dataset},{int(cost_human)},{relative_error},{cost_llm},{cost_human}\n")
        else:
            print(f"Relative error of LLM only inference: {relative_error}")
            print(f"Cost: {cost_llm} llm and {int(cost_human)} human samples")