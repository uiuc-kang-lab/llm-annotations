import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd



DATASETS = [
    "mt-bench", 
    "helmet", 
    "implicit_hate", 
    "med-safe", 
    "mrpc", 
    # "persuasion"
]

METHODS = [
    "llm_only",
    "uniform_sampling",
    "control_variate",
    "importance_sampling",
    # "llm_human",
    # "control_variate_importance_sampling"
]

colors = sns.color_palette("BuPu", n_colors=len(METHODS))

def plot_one_line(x, y, label):
    plt.plot(x, y, label=label, color=colors[METHODS.index(label)])

def read_data_from_csv(dataset, method):
    if not os.path.exists(f"results/{method}/{dataset}_{method}.csv"):
        print(f"File not found: results/{method}/{dataset}_{method}.csv")
        return None, None
    df = pd.read_csv(f"results/{method}/{dataset}_{method}.csv")
    return df['Human_Samples'], df['AvgRelativeError']

def method2label(method):
    if method == "llm_only":
        return "LLM Only"
    elif method == "uniform_sampling":
        return "Uniform Sampling"
    elif method == "llm_human":
        return "Confidence Priority"
    elif method == "importance_sampling":
        return "Importance Sampling"
    elif method == "control_variate":
        return "Control Variate"
    elif method == "control_variate_importance_sampling":
        return "Control Variate + Importance Sampling"

def plot_one_dataset(dataset):
    plt.figure(figsize=(5, 3))
    for method in METHODS:
        x, y = read_data_from_csv(dataset, method)
        if x is not None and y is not None:
            if method == "llm_only":
                plt.axhline(y=y.iloc[0], label=method2label(method))
            else:
                plot_one_line(x, y, label=method)
    plt.xlabel("Human Samples")
    plt.ylabel("Average Relative Error")
    plt.legend()
    plt.savefig(f"plots/{dataset}.pdf", bbox_inches='tight')
    plt.cla()

def main():
    for dataset in DATASETS:
        plot_one_dataset(dataset)

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    main()