import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directories for results
RESULTS_DIR = "/Users/zacharylee/llm-annotations/results"
METHODS = ["importance_sampling", "control_variate"]

# Define datasets to plot
DATASETS = ["global_warming", "helmet", "implicit_hate", "med-safe", "mrpc", "persuasion"]

# Function to load results for a specific method and dataset
def load_results(method, dataset):
    file_path = os.path.join(RESULTS_DIR, method, f"{dataset}_{method}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

# Function to plot results for a specific dataset
def plot_dataset(dataset):
    plt.figure(figsize=(10, 6))

    # Plot results for Importance Sampling and Control Variate
    for method in METHODS:
        results = load_results(method, dataset)
        if results is not None:
            if "Human_Samples" in results.columns:
                x = results["Human_Samples"]
            elif "Human Samples" in results.columns:
                x = results["Human Samples"]
            else:
                print(f"Skipping {method} for {dataset}: No Human Samples column found.")
                continue

            y = results["Relative Error"] if "Relative Error" in results.columns else results["AvgRelativeError"]
            plt.plot(x, y, label=method.replace("_", " ").title())

    plt.title(f"Importance vs Control Variate for {dataset.replace('_', ' ').title()}")
    plt.xlabel("Number of Human Samples")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset}_importance_control_comparison.png"))
    plt.close()

# Main function to plot results for all datasets
def main():
    for dataset in DATASETS:
        print(f"Plotting Importance vs Control Variate for {dataset}...")
        plot_dataset(dataset)
    print("Plots saved in the 'results/plots' directory.")

if __name__ == "__main__":
    main()