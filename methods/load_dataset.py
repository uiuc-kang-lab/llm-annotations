import pandas as pd
from typing import Tuple
from statistic import compute_statistics

def load_data(dataset: str) -> Tuple[pd.DataFrame, float]:
    if dataset == "global_warming":
        data = pd.read_csv("../datasets/global_warming/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)
    elif dataset == "indian_dialect":
        data = pd.read_csv("../datasets/indian_dialect/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
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
    elif dataset == "mathematical_capabilities":
        data = pd.read_csv("/Users/zacharylee/TestTest/llm-annotations/dataset/mathematical_capabilities/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)
    elif dataset == "persuasion":
        data = pd.read_csv("/Users/zacharylee/TestTest/llm-annotations/dataset/persuasion/data.csv")
        data["label"] = data["gold_label"]
        groundtruth = compute_statistics(data, dataset)
        data.drop(columns=["label"], inplace=True)        
    else:
        raise NotImplementedError("Dataset not implemented")
    
    return data, groundtruth