import pandas as pd
from typing import Tuple
from methods.statistic import compute_statistics

def load_data(dataset: str, label_column: str = "gold_label") -> Tuple[pd.DataFrame, float]:
    if dataset == "global_warming":
        data = pd.read_csv("../llm-annotations/datasets/global_warming/global_warming.csv")
    elif dataset == "helmet":
        data = pd.read_csv("../llm-annotations/datasets/helmet/helmet.csv")
        data["proxy_label"] = data.apply(lambda row: row["gpt_confidence"] if row["gpt_label"] == 1 else 1-row["gpt_confidence"], axis=1)
    elif dataset == "implicit_hate":
        data = pd.read_csv("../llm-annotations/datasets/implicit_hate/implicit_hate.csv")
        data["proxy_label"] = data.apply(lambda row: row["Confidence"] if row["Confidence"] == "white_grievance" else 1-row["Confidence"], axis=1)
    elif dataset == "persuasion":
        data = pd.read_csv("../llm-annotations/datasets/persuasion/persuasion.csv")
        data["proxy_label"] = data.apply(lambda row: row["gpt_confidence"] if row["gpt_label"] == "true" else 1-row["gpt_confidence"], axis=1)
    elif dataset == "mrpc":
        data = pd.read_csv("../llm-annotations/datasets/coannotating/MRPC.csv")
        data["proxy_label"] = data.apply(lambda row: row["confidence"] if row["gpt_label"] == 1 else 1-row["confidence"], axis=1)
    elif dataset == "med-safe":
        data = pd.read_csv("../llm-annotations/datasets/judge-bench/judge_bench.csv")
        data["proxy_label"] = data.apply(lambda row: 0.75 if row["gpt_label"] == "Serious" else 0.25, axis=1)
    elif dataset == "mt-bench":
        data = pd.read_csv("../llm-annotations/datasets/mt-bench/mt-bench.csv")
        data["proxy_label"] = data.apply(lambda row: row["confidence"] if (row["gpt_label"] == "model_a" and row["model_a"] == "gpt-3.5-turbo") or (row["gpt_label"] == "model_b" and row["model_b"] == "gpt-3.5-turbo") else 1-row["confidence"], axis=1)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")
    
    statistic = compute_statistics(data, dataset, label_column=label_column)
    return data, statistic