import pandas as pd

def compute_statistics(data: pd.DataFrame, dataset: str, label_column: str = "gpt_label"):
    if dataset == "global_warming":
        n_data_affirming = len(data[data["contains_affirming_device"] == True])
        n_data_not_affirming = len(data[data["contains_affirming_device"] == False])
        n_data_agree_affirming = len(data[(data[label_column] == 1) & (data["contains_affirming_device"] == True)])
        n_data_agree_not_affirming = len(data[(data[label_column] == 1) & (data["contains_affirming_device"] == False)])
        
        if n_data_affirming == 0 or n_data_not_affirming == 0:
            return 0
        
        p_affirming = n_data_agree_affirming / n_data_affirming
        p_not_affirming = n_data_agree_not_affirming / n_data_not_affirming
        
        if p_affirming == 0 or p_affirming == 1 or p_not_affirming == 0 or p_not_affirming == 1:
            return 0

        odds = ((p_affirming) / (1 - p_affirming)) / ((p_not_affirming) / (1 - p_not_affirming))
        return odds

    if dataset == "helmet":
        n_positive = len(data[data[label_column] == 1])
        percentage = n_positive / len(data)
        return percentage

    if dataset == "implicit_hate":
        n_white_grievance = len(data[data[label_column] == "white_grievance"])
        percentage = n_white_grievance / len(data)
        return percentage

    if dataset == "persuasion":
        data[label_column] = data[label_column].astype(str).str.strip().str.lower()
        data["gold_label"] = data["gold_label"].astype(str).str.strip().str.lower()
        n_true = len(data[data[label_column] == "true"])
        prevalence_true = n_true / len(data)
        return prevalence_true

    if dataset == "mrpc":
        n_positive = len(data[data[label_column] == 1])
        proportion = n_positive / len(data)
        return proportion

    if dataset == "med-safe":
        n_serious = len(data[data[label_column] == "Serious"])
        percentage = n_serious / len(data)
        return percentage
    
    if dataset == "mt-bench":
        wins = ((data["model_a"] == "gpt-3.5-turbo") & (data[label_column] == "model_a")) | \
                ((data["model_b"] == "gpt-3.5-turbo") & (data[label_column] == "model_b"))
        return wins.sum() / len(data)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")
    
    
def label2res(label, dataset: str):
    if dataset == "helmet":
        return label
    elif dataset == "implicit_hate":
        return label == "white_grievance"
    elif dataset == "persuasion":
        return label == "true"
    elif dataset == "mrpc":
        return label == 1
    elif dataset == "med-safe":
        return label == "Serious"
    elif dataset == "mt-bench":
        model_a, model_b, label = label
        return (model_a == "gpt-3.5-turbo" and label == "model_a") or (model_b == "gpt-3.5-turbo" and label == "model_b")