import os
import pandas as pd
import numpy as np

RESULTS_DIR = "/Users/zacharylee/llm-annotations/results"
DATASETS = ["helmet", "implicit_hate", "med-safe", "mrpc", "persuasion"]
TARGET_METHODS = ["importance_sampling", "control_variate"]
BASELINES = ["uniform_sampling", "llm_human", "llm_only"]

def get_llm_only_baseline(dataset):
    from methods.llm_only import run_llm_only
    relative_error, _, _ = run_llm_only(dataset)
    return relative_error

def load_results(method, dataset):
    file_path = os.path.join(RESULTS_DIR, method, f"{dataset}_{method}.csv")
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

def get_sample_col(df):
    for col_candidate in ["Human_Samples", "Human Samples", "Cost_Human"]:
        if col_candidate in df.columns:
            return col_candidate
    return None

def get_error_col(df):
    for candidate in ["Relative Error", "AvgRelativeError", "Relative_Error", "RelError"]:
        if candidate in df.columns:
            return candidate
    return None

def compute_csv_to_csv_improvement(baseline_df, target_df):
    if baseline_df is None or target_df is None:
        return None

    baseline_sample = get_sample_col(baseline_df)
    target_sample   = get_sample_col(target_df)
    baseline_error  = get_error_col(baseline_df)
    target_error    = get_error_col(target_df)
    if not (baseline_sample and target_sample and baseline_error and target_error):
        return None

    rename_baseline = {baseline_sample: "SampleCol", baseline_error: "BaselineError"}
    rename_target   = {target_sample: "SampleCol",  target_error:   "TargetError"}

    bdf = baseline_df[[baseline_sample, baseline_error]].rename(columns=rename_baseline)
    tdf = target_df[[target_sample, target_error]].rename(columns=rename_target)

    merged = pd.merge(bdf, tdf, on="SampleCol", how="inner")
    improvements = []
    for _, row in merged.iterrows():
        base_err = row["BaselineError"]
        tgt_err  = row["TargetError"]
        if base_err != 0:
            improvements.append((base_err - tgt_err) / base_err * 100)

    if not improvements:
        return 0.0
    return sum(improvements) / len(improvements)

def compute_llm_only_improvement(llm_only_err, target_df):
    if target_df is None or llm_only_err is None:
        return None

    target_error_col = get_error_col(target_df)
    if not target_error_col or llm_only_err == 0:
        return None

    errs = target_df[target_error_col].dropna()
    if errs.empty:
        return 0.0

    improvements = []
    for tgt_err in errs:
        improvements.append((llm_only_err - tgt_err) / llm_only_err * 100)
    return sum(improvements) / len(improvements)

def main():
    improvements_all = {baseline: {m: [] for m in TARGET_METHODS} for baseline in BASELINES}

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")
        uniform_df = load_results("uniform_sampling", dataset)
        llm_human_df = load_results("llm_human", dataset)
        llm_only_err = get_llm_only_baseline(dataset)

        for method in TARGET_METHODS:
            target_df = load_results(method, dataset)

            # Compare vs uniform_sampling
            improv_unif = compute_csv_to_csv_improvement(uniform_df, target_df)
            if improv_unif is not None:
                print(f"[vs UNIFORM] {method} improves ~{improv_unif:.2f}%")
                improvements_all["uniform_sampling"][method].append(improv_unif)
            else:
                print(f"[vs UNIFORM] No valid comparison for {method}.")

            # Compare vs llm_human
            improv_human = compute_csv_to_csv_improvement(llm_human_df, target_df)
            if improv_human is not None:
                print(f"[vs LLM+HUMAN] {method} improves ~{improv_human:.2f}%")
                improvements_all["llm_human"][method].append(improv_human)
            else:
                print(f"[vs LLM+HUMAN] No valid comparison for {method}.")

            # Compare vs llm_only
            improv_llm_only = compute_llm_only_improvement(llm_only_err, target_df)
            if improv_llm_only is not None:
                print(f"[vs LLM_ONLY] {method} improves ~{improv_llm_only:.2f}%")
                improvements_all["llm_only"][method].append(improv_llm_only)
            else:
                print(f"[vs LLM_ONLY] No valid comparison for {method}.")

    # Print summary of average improvements across all datasets
    print("\n=== AVERAGE IMPROVEMENTS ACROSS ALL DATASETS ===")
    for baseline in BASELINES:
        for method in TARGET_METHODS:
            vals = improvements_all[baseline][method]
            if vals:
                avg_val = sum(vals) / len(vals)
                print(f"{method} vs {baseline}: {avg_val:.2f}% average improvement")
            else:
                print(f"{method} vs {baseline}: No valid results.")

    # Compute average of control_variate & importance_sampling vs each baseline
    print("\n=== AVERAGE OF CONTROL_VARIATE & IMPORTANCE_SAMPLING (BY BASELINE) ===")
    for baseline in BASELINES:
        combined_vals = (
            improvements_all[baseline]["control_variate"]
            + improvements_all[baseline]["importance_sampling"]
        )
        if combined_vals:
            avg_val_methods = sum(combined_vals) / len(combined_vals)
            print(f"Combined vs {baseline}: {avg_val_methods:.2f}%")
        else:
            print(f"No combined results for {baseline}.")

    # Compute overall average for both methods across all baselines
    print("\n=== OVERALL AVERAGE ACROSS ALL BASELINES ===")
    all_improvements = []
    for baseline in BASELINES:
        all_improvements += improvements_all[baseline]["control_variate"]
        all_improvements += improvements_all[baseline]["importance_sampling"]
    if all_improvements:
        overall_avg = sum(all_improvements) / len(all_improvements)
        print(f"Overall average improvement: {overall_avg:.2f}%")
    else:
        print("No overall improvements found.")

def required_sample_size(df, target_err):
    """
    Given a DataFrame (from a results CSV) with columns "Human Samples" and a relative error column,
    return the minimum number of human samples required for the relative error to drop
    below or equal to target_err. If the target is never reached, return the maximum samples.
    """
    sample_col = get_sample_col(df)
    err_col = get_error_col(df)
    if sample_col is None or err_col is None:
        return None
    
    # Sort by sample size (ascending)
    sorted_df = df.sort_values(by=sample_col)
    # Find rows where the relative error is below the target
    target_rows = sorted_df[sorted_df[err_col] <= target_err]
    if not target_rows.empty:
        return target_rows[sample_col].iloc[0]
    else:
        # If the target error is never reached, use the maximum sample size as a fallback
        return sorted_df[sample_col].max()

# Set the desired target error threshold (in the same units as in the CSV, e.g., percentage)
target_error_threshold = .1  # e.g., we want the relative error to be at most 10%

# We will compute cost savings comparing the uniform sampling baseline (N_uniform)
# against each target method (e.g., control variates and importance sampling)
print("\n=== ANNOTATION COST SAVINGS (AT TARGET ERROR THRESHOLD) ===")
cost_savings = {method: [] for method in TARGET_METHODS}
for dataset in DATASETS:
    uniform_df = load_results("uniform_sampling", dataset)
    if uniform_df is None:
        print(f"No uniform sampling results for {dataset}. Skipping.")
        continue

    N_uniform = required_sample_size(uniform_df, target_error_threshold)
    if N_uniform is None:
        print(f"Could not determine required samples from uniform sampling for {dataset}.")
        continue

    print(f"\nDataset: {dataset} (Uniform requires {N_uniform} samples for error ≤ {target_error_threshold}%)")
    for method in TARGET_METHODS:
        target_df = load_results(method, dataset)
        if target_df is None:
            print(f"  No results for {method} on {dataset}.")
            continue

        N_method = required_sample_size(target_df, target_error_threshold)
        if N_method is None:
            print(f"  Could not determine required samples for {method} on {dataset}.")
            continue
        
        # Compute relative savings: How many fewer labels are needed relative to uniform?
        savings = (N_uniform - N_method) / N_uniform * 100  # percentage savings
        improvement_factor = N_uniform / N_method  # cost improvement factor
        cost_savings[method].append(savings)
        print(f"  {method}: {N_method} samples required. Savings: {savings:.2f}% (Improvement factor: {improvement_factor:.2f}x)")

# Compute cost savings comparing the uniform sampling baseline (N_uniform)
# against each target method (e.g., control variates and importance sampling)
print("\n=== ANNOTATION COST SAVINGS (AT TARGET ERROR THRESHOLD) ===")
cost_savings = {method: [] for method in TARGET_METHODS}
improvement_factors = {method: [] for method in TARGET_METHODS}   # new dict for improvement multipliers
for dataset in DATASETS:
    uniform_df = load_results("uniform_sampling", dataset)
    if uniform_df is None:
        print(f"No uniform sampling results for {dataset}. Skipping.")
        continue

    N_uniform = required_sample_size(uniform_df, target_error_threshold)
    if N_uniform is None or N_uniform == 0:
        print(f"Dataset: {dataset} returned {N_uniform} samples for uniform sampling. Skipping cost savings computation.")
        continue

    print(f"\nDataset: {dataset} (Uniform requires {N_uniform} samples for error ≤ {target_error_threshold}%)")
    for method in TARGET_METHODS:
        target_df = load_results(method, dataset)
        if target_df is None:
            print(f"  No results for {method} on {dataset}.")
            continue

        N_method = required_sample_size(target_df, target_error_threshold)
        if N_method is None:
            print(f"  Could not determine required samples for {method} on {dataset}.")
            continue
        
        # Compute relative savings: How many fewer labels are needed compared to uniform?
        savings = (N_uniform - N_method) / N_uniform * 100  # percentage savings
        improvement_factor = N_uniform / N_method if N_method != 0 else float('inf')  # cost improvement multiplier
        cost_savings[method].append(savings)
        improvement_factors[method].append(improvement_factor)
        print(f"  {method}: {N_method} samples required. Savings: {savings:.2f}% (Improvement factor: {improvement_factor:.2f}x)")

# Report average cost savings per target method across datasets
print("\n=== AVERAGE COST SAVINGS ACROSS DATASETS ===")
for method, savings_list in cost_savings.items():
    if savings_list:
        avg_savings = sum(savings_list) / len(savings_list)
        print(f"{method}: {avg_savings:.2f}% average savings")
    else:
        print(f"{method}: No valid savings computed.")

# --- New Block: Compute and report overall average improvement multiplier ---
print("\n=== AVERAGE IMPROVEMENT MULTIPLIER ACROSS DATASETS ===")
for method, factors in improvement_factors.items():
    if factors:
        avg_factor = sum(factors) / len(factors)
        print(f"{method}: {avg_factor:.2f}x average improvement multiplier")
    else:
        print(f"{method}: No improvement multiplier data.")

# Define a list of thresholds (e.g., 20 points between 1 and 0)
thresholds = np.linspace(1.0, 0.0, num=20)

# Create containers to accumulate savings and multipliers for each method, across thresholds and datasets.
savings_across_thresholds = {method: [] for method in TARGET_METHODS}
multipliers_across_thresholds = {method: [] for method in TARGET_METHODS}

# Iterate over error thresholds
for thr in thresholds:
    # For each threshold, accumulate savings and improvement multipliers across datasets.
    current_threshold_savings = {method: [] for method in TARGET_METHODS}
    current_threshold_multipliers = {method: [] for method in TARGET_METHODS}
    for dataset in DATASETS:
        uniform_df = load_results("uniform_sampling", dataset)
        if uniform_df is None:
            continue
        N_uniform = required_sample_size(uniform_df, thr)
        if N_uniform is None or N_uniform == 0:
            continue

        for method in TARGET_METHODS:
            target_df = load_results(method, dataset)
            if target_df is None:
                continue
            N_method = required_sample_size(target_df, thr)
            if N_method is None or N_method == 0:
                continue

            # Calculate cost savings and improvement multiplier for this dataset at threshold thr.
            savings = (N_uniform - N_method) / N_uniform * 100  # percentage savings
            multiplier = N_uniform / N_method  # improvement multiplier
            current_threshold_savings[method].append(savings)
            current_threshold_multipliers[method].append(multiplier)
    
    # Average the savings and multipliers at this threshold for each method.
    for method in TARGET_METHODS:
        if current_threshold_savings[method]:
            avg_savings_thr = sum(current_threshold_savings[method]) / len(current_threshold_savings[method])
            savings_across_thresholds[method].append(avg_savings_thr)
        if current_threshold_multipliers[method]:
            avg_multiplier_thr = sum(current_threshold_multipliers[method]) / len(current_threshold_multipliers[method])
            multipliers_across_thresholds[method].append(avg_multiplier_thr)

# Now compute the overall average across thresholds for each method.
print("\n=== OVERALL AVERAGE COST SAVINGS ACROSS THRESHOLDS ===")
for method, savings_list in savings_across_thresholds.items():
    if savings_list:
        overall_avg_savings = sum(savings_list) / len(savings_list)
        print(f"{method}: {overall_avg_savings:.2f}% average savings (averaged over thresholds)")
    else:
        print(f"{method}: No valid savings data.")

print("\n=== OVERALL AVERAGE IMPROVEMENT MULTIPLIER ACROSS THRESHOLDS ===")
for method, multipliers_list in multipliers_across_thresholds.items():
    if multipliers_list:
        overall_avg_multiplier = sum(multipliers_list) / len(multipliers_list)
        print(f"{method}: {overall_avg_multiplier:.2f}x average improvement multiplier (averaged over thresholds)")
    else:
        print(f"{method}: No valid multiplier data.")

# --- New Block: Compute and print average relative error for the med-safe dataset
print("\n=== RELATIVE ERROR FOR MED-SAFE DATASET ===")
dataset = "med-safe"
# Create a list that includes TARGET_METHODS, "llm_human", and "uniform_sampling"
methods_to_check = TARGET_METHODS + ["llm_human", "uniform_sampling"]
for method in methods_to_check:
    df_med_safe = load_results(method, dataset)
    if df_med_safe is None:
        print(f"No results for {method} on {dataset}.")
        continue
    err_col = get_error_col(df_med_safe)
    if err_col is None:
        print(f"No relative error column found for {method} on {dataset}.")
        continue
    
    # Compute average relative error from the relative error values in the CSV
    avg_rel_error = np.mean(df_med_safe[err_col])
    print(f"{method}: Average Relative Error = {avg_rel_error * 100:.4f}%")

if __name__ == "__main__":
    # Example usage: print out minimum samples for uniform sampling on a specific dataset.
    dataset = "med-safe"
    uniform_df = load_results("uniform_sampling", dataset)
    if uniform_df is not None:
        target_err = target_error_threshold  # using the defined threshold (e.g., 0.1)
        min_samples = required_sample_size(uniform_df, target_err)
        print(f"\nFor dataset '{dataset}', the uniform sampling method requires {min_samples} samples to achieve ≤ {target_err}% error.")
    else:
        print(f"No uniform sampling results for {dataset}.")
    
    main()