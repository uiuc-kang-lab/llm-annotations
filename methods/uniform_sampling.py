import sys
import numpy as np
from load_dataset import load_data
from statistic import compute_statistics
import argparse

def run_uniform_sampling(dataset: str, step_size: int, repeat: int):
    """
    Run uniform sampling by incrementally replacing LLM samples with human samples.

    Args:
        dataset (str): The name of the dataset.
        step_size (int): The number of samples to increment at each step.
        repeat (int): The number of iterations for each step size.

    Returns:
        None
    """
    # Load the dataset and compute the ground truth
    data, groundtruth = load_data(dataset)
    total_size = len(data)
    
    # Start with all samples labeled by GPT
    for human_samples in range(0, total_size + 1, step_size):
        # Calculate the number of LLM samples
        llm_samples = total_size - human_samples
        
        # Store relative errors for multiple iterations
        relative_errors = []
        
        for _ in range(repeat):
            # Shuffle the dataset for each iteration
            sampled_data = data.sample(frac=1, random_state=None).reset_index(drop=True)
            
            # Construct the label column
            sampled_data["label"] = sampled_data["gpt_label"].to_numpy()
            if human_samples > 0:
                sampled_data.loc[:human_samples - 1, "label"] = sampled_data["gold_label"].to_numpy()[:human_samples]
            
            # Compute the statistic for the hybrid dataset
            try:
                estimate = compute_statistics(sampled_data, dataset, label_column="label")
            except Exception as e:
                print(f"Error computing statistics: {e}")
                estimate = 0
            
            # Compute the relative error
            relative_error = abs(estimate - groundtruth) / groundtruth if groundtruth != 0 else float('inf')
            relative_errors.append(relative_error)
        
        # Compute the average relative error across iterations
        avg_relative_error = np.mean(relative_errors)
        
        # Print the results
        print(f"Average relative error of uniform sampling: {avg_relative_error}")
        print(f"Cost: {llm_samples} llm and {human_samples} human samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run uniform sampling on a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--step_size', type=int, default=100, help='Number of samples to increment at each step')
    parser.add_argument('--repeat', type=int, default=10, help='Number of iterations for each step size')
    args = parser.parse_args()
    
    # Run uniform sampling
    run_uniform_sampling(args.dataset, step_size=args.step_size, repeat=args.repeat)