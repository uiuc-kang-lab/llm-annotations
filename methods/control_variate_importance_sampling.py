import numpy as np
import pandas as pd
from methods.load_dataset import load_data
from methods.statistic import compute_statistics
import argparse
import os

def control_variate_importance_sampling(
    df: pd.DataFrame,
    confidence_col: str,
    gold_label_col: str,
    gpt_label_col: str,
    sample_sizes: list,
    repeat: int = 1000,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Perform control variate importance sampling using model confidence.
    
    Combines Importance Sampling (IS) with Control Variate (CV) correction:
    1. Draw samples with probabilities ∝ (1 - confidence) (Importance Sampling)
    2. Compute Horvitz-Thompson estimate of the target statistic
    3. Apply control variate adjustment using confidence as proxy variable
    
    Args:
        df: Input dataframe with confidence scores and labels
        confidence_col: Column name for confidence scores
        gold_label_col: Column name for gold standard labels
        gpt_label_col: Column name for GPT predicted labels
        sample_sizes: List of sample sizes to evaluate
        repeat: Number of repetitions for stability
        dataset_name: Name of dataset for statistic computation
        
    Returns:
        DataFrame with Human Samples and Relative Error columns
    """
    # Compute true statistic using full dataset with GPT labels
    true_statistic = compute_statistics(df, dataset_name, label_column=gpt_label_col)
    
    # Compute full population mean for control variate
    t_full = df[confidence_col].mean()
    
    # Importance sampling weights: proportional to (1 - confidence)
    # Higher weights for lower confidence samples
    importance_weights = (1 - df[confidence_col]) / (1 - df[confidence_col]).sum()
    
    results = {"Human Samples": [], "Relative Error": []}
    
    for n_samples in sample_sizes:
        errors = []
        
        for _ in range(repeat):
            # Sample with importance weights (proportional to 1 - confidence)
            sampled_indices = np.random.choice(
                df.index, 
                size=n_samples, 
                p=importance_weights.values, 
                replace=True
            )
            sampled = df.loc[sampled_indices].copy()
            
            # Compute sampling probabilities for Horvitz-Thompson estimation
            sampling_probs = importance_weights.loc[sampled_indices]
            
            # Horvitz-Thompson estimate of the statistic
            # For each sampled point, weight by 1/sampling_probability
            ht_weights = 1.0 / (n_samples * sampling_probs)
            
            # Compute base estimate using GPT labels (to match true statistic)
            estimate = compute_statistics(sampled, dataset_name, label_column=gpt_label_col)
            
            # Control variate adjustment using confidence as proxy
            t_hat = sampled[confidence_col].mean()
            
            # Compute control variate coefficient
            correctness = (sampled[gold_label_col] == sampled[gpt_label_col]).astype(float)
            cov = np.cov(sampled[confidence_col], correctness, ddof=0)[0, 1]
            var_t = np.var(sampled[confidence_col], ddof=0)
            c_hat = -cov / var_t if var_t > 1e-6 else 0
            
            # Apply control variate correction
            adjusted_estimate = estimate + c_hat * (t_hat - t_full)
            
            # Compute relative error
            error = abs(adjusted_estimate - true_statistic) / true_statistic if true_statistic != 0 else float('inf')
            errors.append(error)
        
        # Store RMSE for this sample size
        results["Human Samples"].append(n_samples)
        results["Relative Error"].append(np.sqrt(np.mean(np.array(errors) ** 2)))
    
    return pd.DataFrame(results)

def run_control_variate_importance_sampling(
    dataset_name: str, 
    step_size: int, 
    repeat: int, 
    save_dir: str
):
    """
    Run control variate importance sampling on a dataset and save results.
    
    Args:
        dataset_name: Name of the dataset
        step_size: Step size for human budgets
        repeat: Number of repetitions for stability
        save_dir: Directory to save results
    """
    # Load dataset
    dataset, _ = load_data(dataset_name)
    
    # Standard column names
    confidence_col = "confidence_normalized"
    gold_label_col = "gold_label"
    gpt_label_col = "gpt_label"
    
    # Determine sample sizes up to full dataset size
    total_samples = len(dataset)
    sample_sizes = list(range(step_size, total_samples + 1, step_size))
    
    # Run the sampling method
    results = control_variate_importance_sampling(
        df=dataset,
        confidence_col=confidence_col,
        gold_label_col=gold_label_col,
        gpt_label_col=gpt_label_col,
        sample_sizes=sample_sizes,
        repeat=repeat,
        dataset_name=dataset_name
    )
    
    # Save results to CSV
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset_name}_control_variate_importance_sampling.csv")
    
    # Add dataset column for consistency with other methods
    results.insert(0, "Dataset", dataset_name)
    results.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control variate importance sampling on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (as defined in load_data)")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for human budgets")
    parser.add_argument("--repeat", type=int, default=1000, help="Number of iterations for stability")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the result")
    args = parser.parse_args()
    
    run_control_variate_importance_sampling(
        dataset_name=args.dataset,
        step_size=args.step_size,
        repeat=args.repeat,
        save_dir=args.save_dir
    )