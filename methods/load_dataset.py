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
    else:
        raise NotImplementedError("Dataset not implemented")
    
    return data, groundtruth