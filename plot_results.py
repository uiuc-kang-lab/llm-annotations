import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from methods.load_dataset import load_data
from methods.statistic import compute_statistics

RESULTS_DIR = "/Users/zacharylee/llm-annotations/results"
METHODS = ["uniform_sampling", "importance_sampling", "control_variate", "llm_human", "control_variate_importance_sampling"]

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

def plot_confidence_vs_correctness(csv_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    df = pd.read_csv(csv_path)
    df['correct'] = df['gpt_label'] == df['gold_label']
    median_conf = df['confidence_normalized'].median()
    lower = df[df['confidence_normalized'] <= median_conf]
    upper = df[df['confidence_normalized'] > median_conf]
    lower_frac = lower['correct'].mean()
    upper_frac = upper['correct'].mean()

    # Plot 1: Bar chart for each group
    plt.figure(figsize=(8, 6))
    plt.bar(['Lower 50% Confidence', 'Upper 50% Confidence'], [lower_frac, upper_frac], color=['orange', 'blue'])
    plt.ylabel("Likelihood of Correctness")
    plt.title("Correctness by Confidence Percentile (Judge-Bench)")
    plt.ylim(0, 1)
    for i, v in enumerate([lower_frac, upper_frac]):
        plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=14)
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "judge_bench_upper_vs_lower_confidence.pdf"), format="pdf")
    plt.close()

    # Plot 2: Difference in correctness
    diff = upper_frac - lower_frac
    plt.figure(figsize=(6, 6))
    plt.bar(['Upper - Lower'], [diff], color='purple')
    plt.ylabel("Difference in Correctness")
    plt.title("Difference in Correctness\n(Upper 50% - Lower 50% Confidence)")
    plt.ylim(-1, 1)
    plt.text(0, diff + 0.02 if diff > 0 else diff - 0.08, f"{diff*100:.1f}%", ha='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "judge_bench_confidence_correctness_difference.pdf"), format="pdf")
    plt.close()

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
                        label=(
                            "Importance + Control Variate" if (dataset == "implicit_hate" and method == "control_variate_importance_sampling")
                            else format_label(method) if (
                                (dataset == "helmet" and method in ["uniform_sampling"]) or
                                (dataset == "implicit_hate" and method in ["importance_sampling", "control_variate"])
                            )
                            else None
                        )
)
    # Reduce the number of ticks for cleaner increments
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))  # Fewer x-axis ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))  # Fewer y-axis ticks

    # Apply percentage formatter to y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    plt.xlabel("Number of Human Samples", fontsize=50)  # Unbolded x-axis label
    plt.ylabel("Relative Error (%)", fontsize=50)  # Updated y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show legend for specific datasets
    if dataset in ["helmet", "implicit_hate"]:
        plt.legend(fontsize=40, loc="best")  # Legend for specific datasets

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
    # NEW: Also plot confidence vs correctness for judge-bench
    plot_confidence_vs_correctness("datasets/judge-bench/judge_bench.csv")
    print("Judge-bench confidence vs correctness plot saved.")

if __name__ == "__main__":
    main()
