import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
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

def format_label(label):
    """Format legend labels by removing underscores, capitalizing words, and replacing 'llm' with 'LLM'."""
    formatted_label = label.replace("_", " ").title().replace("Llm", "LLM")
    if formatted_label == "LLM Human":
        return "LLM + Human"  # Special case for LLM Human
    return formatted_label

def percentage_formatter(x, _):
    """Format y-axis labels as plain numbers multiplied by 100."""
    return f"{x * 100:.0f}"  # Removed the '%' symbol

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
        label=format_label("llm_only_baseline") if dataset == "helmet" else None  # Add to legend only for "helmet"
    )

    # Plot results for each method
    for method in METHODS:
        results = load_results(method, dataset)
        if results is not None:
            if method == "llm_human":
                if "Cost_Human" in results.columns and "Relative_Error" in results.columns:
                    x = results["Cost_Human"]
                    y = results["Relative_Error"]
                    plt.plot(
                        x, y, linestyle="-", color="purple", linewidth=6,
                        label=format_label("llm_human") if dataset == "helmet" else None  # Add to legend only for "helmet"
                    )
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
                    plt.plot(
                        x, y, linewidth=6,
                        label=format_label(method) if (dataset == "helmet" and method in ["uniform_sampling"]) or
                                                     (dataset == "implicit_hate" and method in ["importance_sampling", "control_variate"])
                                                     else None  # Add to legend conditionally
                    )

    # Reduce the number of ticks for cleaner increments
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))  # Fewer x-axis ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))  # Fewer y-axis ticks

    # Apply percentage formatter to y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    plt.xlabel("Number of Human Samples", fontsize=50)  # Unbolded x-axis label
    plt.ylabel("Relative Error (x100)", fontsize=50)  # Updated y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show legend for specific datasets
    if dataset in ["helmet", "implicit_hate"]:
        plt.legend(fontsize=50, loc="best")  # Legend for specific datasets

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