import pandas as pd
from typing import Tuple
from statistic import compute_statistics

def load_data(dataset: str) -> Tuple[pd.DataFrame, float]:
    if dataset == "global_warming":
        data = pd.read_csv("../datasets/global_warming/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)
        data.drop(columns=["label"], inplace=True)
    elif dataset == "helmet":
        data = pd.read_csv("../datasets/helmet/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)
    elif dataset == "implicit_hate":
        data = pd.read_csv("../dataset/implicit_hate/formatted_data_gpt_results.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)
    elif dataset == "persuasion":
        data = pd.read_csv("/Users/zacharylee/TestTest/llm-annotations/dataset/persuasion/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)
    elif dataset == "mrpc":
        data = pd.read_csv("/Users/zacharylee/TestTest/llm-annotations/coannotating_datasets/MRPC.csv")
        groundtruth = compute_statistics(data, dataset)   
    elif dataset == "med-safe":
        data = pd.read_csv("judge-bench/full_medical_safety_data.csv")
        groundtruth = compute_statistics(data, dataset)   
    else:
        raise NotImplementedError("Dataset not implemented")
    
    return data, groundtruth