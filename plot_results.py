import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from methods.load_dataset import load_data
from methods.statistic import compute_statistics

RESULTS_DIR = "/Users/zacharylee/llm-annotations/results"
METHODS = ["uniform_sampling", "importance_sampling", "control_variate", "llm_human"]

DATASETS = ["global_warming", "helmet", "implicit_hate", "med-safe", "mrpc", "persuasion"]

def load_results(method, dataset):
    file_path = os.path.join(RESULTS_DIR, method, f"{dataset}_{method}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

def get_llm_only_baseline(dataset):
    from methods.llm_only import run_llm_only  
    relative_error, _, _ = run_llm_only(dataset)
    return relative_error

def plot_dataset(dataset):
    # Bigger figure and thicker axis lines
    plt.figure(figsize=(18, 14))  # Larger figure size for better readability
    plt.rc('axes', linewidth=5)  # Thicker axis lines
    plt.tick_params(axis='both', which='major', labelsize=50)  # MASSIVE tick labels

    # Add the baseline for LLM Only with a visible label
    llm_only_baseline = get_llm_only_baseline(dataset)
    plt.axhline(
        y=llm_only_baseline,
        color="red",
        linestyle="--",
        linewidth=6,  # Thicker baseline line
        label="LLM Only Baseline"
    )

    # Plot results for each method
    for method in METHODS:
        results = load_results(method, dataset)
        if results is not None:
            if method == "llm_human":
                if "Cost_Human" in results.columns and "Relative_Error" in results.columns:
                    x = results["Cost_Human"]
                    y = results["Relative_Error"]
                    plt.plot(x, y, linestyle="-", color="purple", linewidth=6, label="LLM + Human")  # Thicker trend line
            else:
                if "Human_Samples" in results.columns:
                    x = results["Human_Samples"]
                elif "Human Samples" in results.columns:
                    x = results["Human Samples"]
                else:
                    continue

                y_col = "Relative Error" if "Relative Error" in results.columns else "AvgRelativeError"
                if y_col in results.columns:
                    y = results[y_col]
                    plt.plot(x, y, linewidth=6, label=method)  # Thicker trend line

    # Reduce the number of ticks for cleaner increments
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))  # Fewer x-axis ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))  # Fewer y-axis ticks

    plt.xlabel("Number of Human Samples", fontsize=50, weight='bold')  # MASSIVE x-axis label
    plt.ylabel("Relative Error", fontsize=50, weight='bold')  # MASSIVE y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show legend only for the "helmet" dataset with larger font size
    if dataset == "helmet":
        plt.legend(fontsize=40, loc="best")  # Increased legend font size

    plt.tight_layout()

    # Save the plot as a PDF
    output_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset}_performance_comparison.pdf"), format="pdf")
    plt.close()

def main():
    for dataset in DATASETS:
        print(f"Plotting results for {dataset}...")
        plot_dataset(dataset)
    print("Plots saved in the 'results/plots' directory as PDF files.")

if __name__ == "__main__":
    main()