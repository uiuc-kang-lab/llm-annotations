import pandas as pd
from typing import Tuple
from methods.statistic import compute_statistics

def load_data(dataset: str, label_column: str = "gold_label") -> Tuple[pd.DataFrame, float]:
    if dataset == "global_warming":
        data = pd.read_csv("../llm-annotations/datasets/global_warming/global_warming.csv")
    elif dataset == "helmet":
        data = pd.read_csv("../llm-annotations/datasets/helmet/helmet.csv")
    elif dataset == "implicit_hate":
        data = pd.read_csv("../llm-annotations/datasets/implicit_hate/implicit_hate.csv")
    elif dataset == "persuasion":
        data = pd.read_csv("../llm-annotations/datasets/persuasion/persuasion.csv")
    elif dataset == "mrpc":
        data = pd.read_csv("../llm-annotations/datasets/coannotating/MRPC.csv")
    elif dataset == "med-safe":
        data = pd.read_csv("../llm-annotations/datasets/judge-bench/judge_bench.csv")
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")
    
    statistic = compute_statistics(data, dataset, label_column=label_column)
    return data, statistic