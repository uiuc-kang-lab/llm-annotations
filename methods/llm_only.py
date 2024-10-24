import sys
from load_dataset import load_data
from statistic import compute_statistics

def run_llm_only(dataset: str):
    data, groundtruth = load_data(dataset)
    data["label"] = data["gpt_label"]
    estimate = compute_statistics(data, dataset)
    relative_error = abs(estimate - groundtruth) / groundtruth
    return relative_error, len(data), 0
    
if __name__ == "__main__":
    dataset = sys.argv[1]
    relative_error, cost_llm, cost_human = run_llm_only(dataset)
    print(f"Relative error of LLM only inference: {relative_error}")
    print(f"Cost: {cost_llm} llm and {cost_human} human")