import pandas as pd
import numpy as np

# Set the confidence threshold we care about
target_conf = 0.0004853190973064

# Load the CSV (adjust the path if necessary)
# Use a single-character comment to ignore comment lines
df = pd.read_csv("datasets/judge-bench/judge_bench.csv", comment='/')

# Filter rows where confidence_normalized is approximately the target value
# Allowing for floating point precision issues
filtered = df[np.isclose(df["confidence_normalized"], target_conf, atol=1e-10)]

if len(filtered) == 0:
    print("No rows found with the specified confidence value.")
else:
    # For med-safe statistic, compute percentage of "Serious" in each label column
    # using "gold_label" for gold and "gpt_label" for GPT predictions.
    def compute_serious_percentage(data, label_column):
        n_serious = len(data[data[label_column] == "Serious"])
        return n_serious / len(data)
    
    gold_percentage = compute_serious_percentage(filtered, "gold_label")
    gpt_percentage = compute_serious_percentage(filtered, "gpt_label")
    
    error = abs(gold_percentage - gpt_percentage)
    
    mse = (gold_percentage - gpt_percentage) ** 2

    print("Filtered rows:", len(filtered))
    print(f"Gold Serious Percentage: {gold_percentage*100:.2f}%")
    print(f"GPT Serious Percentage: {gpt_percentage*100:.2f}%")
    print(f"Absolute Error: {error*100:.2f}%")
    print(f"MSE: {mse:.6f}")