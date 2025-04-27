import os
import pandas as pd
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(12, 8))  # Increased figure size for better legibility

    # Add the baseline for LLM Only
    llm_only_baseline = get_llm_only_baseline(dataset)
    plt.axhline(y=llm_only_baseline, color="red", linestyle="--")  # Removed label for legend

    # Plot results for each method
    for method in METHODS:
        results = load_results(method, dataset)
        if results is not None:
            if method == "llm_human":
                # Handle LLM + Human results specifically
                if "Cost_Human" in results.columns and "Relative_Error" in results.columns:
                    x = results["Cost_Human"]
                    y = results["Relative_Error"]
                    plt.plot(x, y, linestyle="-", color="purple")  # Removed label for legend
                else:
                    print(f"LLM + Human results for {dataset} are missing required columns.")
            else:
                # Handle other methods
                if "Human_Samples" in results.columns:
                    x = results["Human_Samples"]
                elif "Human Samples" in results.columns:
                    x = results["Human Samples"]
                else:
                    continue  # Skip if no human samples column is found

                y = results["Relative Error"] if "Relative Error" in results.columns else results["AvgRelativeError"]
                plt.plot(x, y)  # Removed label for legend

    plt.xlabel("Number of Human Samples", fontsize=14)
    plt.ylabel("Relative Error", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PDF
    output_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset}_performance_comparison.pdf"), format="pdf")
    plt.close()

# Main function to plot results for all datasets
def main():
    for dataset in DATASETS:
        print(f"Plotting results for {dataset}...")
        plot_dataset(dataset)
    print("Plots saved in the 'results/plots' directory as PDF files.")

if __name__ == "__main__":
    main()