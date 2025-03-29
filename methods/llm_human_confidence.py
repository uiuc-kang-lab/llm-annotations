import sys
import numpy as np
from load_dataset import load_data
from statistic import compute_statistics
import argparse
import os

def run_llm_human_confidence(dataset: str, human_budget: float, use_confidence: bool=False, repeat: int=1000):
    data, groundtruth = load_data(dataset)
    
    human_label_size = int(human_budget * len(data))
    human_label_size = min(human_label_size, len(data))  # Ensure it does not exceed dataset size

    if use_confidence:
        data.sort_values(by="confidence_normalized", ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data["label"] = data["gold_label"].to_numpy()

        if human_label_size < len(data):
            data.iloc[human_label_size:, data.columns.get_loc("label")] = \
                data.iloc[human_label_size:, data.columns.get_loc("gpt_label")].to_numpy()
        try:
            estimate = compute_statistics(data, dataset, label_column="label")
        except Exception as e:
            print(f"Error computing statistics: {e}")
            estimate = 0
            return None, len(data), human_label_size
        relative_error = abs(estimate - groundtruth) / groundtruth

    else:
        relative_errors = []
        for _ in range(repeat):
            shuffled = data.sample(frac=1).reset_index(drop=True)  # keep original data intact
            shuffled["label"] = shuffled["gold_label"].to_numpy()
            if human_label_size < len(shuffled):
                shuffled.iloc[human_label_size:, shuffled.columns.get_loc("label")] = \
                    shuffled.iloc[human_label_size:, shuffled.columns.get_loc("gpt_label")].to_numpy()
            try:
                estimate = compute_statistics(shuffled, dataset, label_column="label")
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
    parser.add_argument('--save_dir', type=str, default="./", help='Directory to save the result')
    args = parser.parse_args()
    args.use_confidence = (args.use_confidence == "True")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    output_file = os.path.join(args.save_dir, f"{args.dataset}_llm_human.csv")
    
    with open(output_file, "w") as f:
        f.write("Dataset,Cost_Human,Relative_Error,Cost_LLM\n")
    
    if args.budget == "all":
        budgets = [0.001 * i for i in range(1, 10)] + \
                  [0.01 * i for i in range(1, 10)] + [0.1 * i for i in range(1, 10)] + [1]
        for budget in budgets:
            relative_error, cost_llm, cost_human = run_llm_human_confidence(
                args.dataset,
                human_budget=float(budget),
                use_confidence=args.use_confidence
            )
            with open(output_file, "a") as f:
                f.write(f"{args.dataset},{int(cost_human)},{relative_error},{cost_llm}\n")
    else:
        relative_error, cost_llm, cost_human = run_llm_human_confidence(
            args.dataset,
            human_budget=float(args.budget),
            use_confidence=args.use_confidence
        )
        with open(output_file, "a") as f:
            f.write(f"{args.dataset},{int(cost_human)},{relative_error},{cost_llm}\n")