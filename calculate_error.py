import os
import pandas as pd

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

if __name__ == "__main__":
    main()