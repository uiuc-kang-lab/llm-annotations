import pandas as pd

def compute_statistics(data: pd.DataFrame, dataset: str):
    if dataset == "global_warming":
        # Proportion of samples with affirming devices
        n_affirming = len(data[data["contains_affirming_device"] == True])
        ground_truth = n_affirming / len(data)
        return ground_truth

    if dataset == "helmet":
        # Proportion of positive labels in the gold standard
        n_positive = len(data[data["gold_label"] == 1])
        ground_truth = n_positive / len(data)
        return ground_truth

    if dataset == "implicit_hate":
        # Proportion of samples labeled as "white_grievance" in the gold standard
        n_white_grievance = len(data[data["gold_label"] == "white_grievance"])
        ground_truth = n_white_grievance / len(data)
        return ground_truth

    if dataset == "persuasion":
        # Proportion of samples labeled as "True" in the gold standard
        n_positive = len(data[data["gold_label"] == "True"])
        ground_truth = n_positive / len(data)
        return ground_truth

    if dataset == "mrpc":
        # Accuracy of human-labeled baseline (if available)
        correct_predictions = (data["gold_label"] == data["gold_label"]).sum()
        total_samples = len(data)
        ground_truth = correct_predictions / total_samples
        return ground_truth

    if dataset == "med-safe":
        # Proportion of samples labeled as "Serious" in the gold standard
        n_serious = len(data[data["gold_label"] == "Serious"])
        ground_truth = n_serious / len(data)
        return ground_truth

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")