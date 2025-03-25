import pandas as pd
from typing import Tuple
from statistic import compute_ground_truth

def load_data(dataset: str) -> Tuple[pd.DataFrame, float]:
    if dataset == "global_warming":
        data = pd.read_csv("../datasets/global_warming/data.csv")
        groundtruth = compute_ground_truth(data, dataset)
    elif dataset == "helmet":
        data = pd.read_csv("../datasets/helmet/data.csv")
        groundtruth = compute_ground_truth(data, dataset)
    elif dataset == "implicit_hate":
        data = pd.read_csv("../dataset/implicit_hate/formatted_data_gpt_results.csv")
        groundtruth = compute_ground_truth(data, dataset)
    elif dataset == "persuasion":
        data = pd.read_csv("/Users/zacharylee/TestTest/llm-annotations/dataset/persuasion/data.csv")
        groundtruth = compute_ground_truth(data, dataset)
    elif dataset == "mrpc":
        data = pd.read_csv("/Users/zacharylee/TestTest/llm-annotations/coannotating_datasets/MRPC.csv")
        groundtruth = compute_ground_truth(data, dataset)
    elif dataset == "med-safe":
        data = pd.read_csv("judge-bench/full_medical_safety_data.csv")
        groundtruth = compute_ground_truth(data, dataset)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")
    
    return data, groundtruth