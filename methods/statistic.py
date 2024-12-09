import pandas as pd

def compute_statistics(data: pd.DataFrame, dataset: str):
    if dataset == "global_warming":
        n_data_affirming = len(data[data["contains_affirming_device"] == True])
        n_data_not_affirming = len(data[data["contains_affirming_device"] == False])
        n_data_agree_affirming = len(data[(data["label"] == 1) & (data["contains_affirming_device"] == True)])
        n_data_agree_not_affirming = len(data[(data["label"] == 1) & (data["contains_affirming_device"] == False)])
        # print(n_data_agree_affirming, n_data_affirming, n_data_agree_not_affirming, n_data_not_affirming)
        odds = ((n_data_agree_affirming / n_data_affirming) / (1 - n_data_agree_affirming / n_data_affirming)) / \
               ((n_data_agree_not_affirming / n_data_not_affirming) / (1 - n_data_agree_not_affirming / n_data_not_affirming))
        return odds
    if dataset == "indian_dialect":
        n_positive = len(data[data["label"] == 3])
        percentage = n_positive / len(data)
        return percentage
    if dataset == "helmet":
        n_positive = len(data[data["label"] == 1])
        percentage = n_positive / len(data)
        return percentage
    if dataset == "implicit_hate":
        n_white_grievance = len(data[data["label"] == "white_grievance"])
        percentage = n_white_grievance / len(data)
        return percentage
    if dataset == "mathematical_capabilities":
        n_positive = len(data[data["gpt_label"] == "success"])
        percentage = n_positive / len(data)
        return percentage
    if dataset == "persuasion":
        n_positive = len(data[data["label"] == "True"])
        percentage = n_positive / len(data)
        return percentage
    else:
        raise NotImplementedError("Dataset not implemented")
