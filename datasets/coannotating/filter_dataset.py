import pandas as pd
import sys

def filter_columns(input_file, output_file):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Keep only the required columns
    required_columns = ['idx', 'gpt_label', 'confidence', 'gold_label']
    df_filtered = df[required_columns]
    
    # Save the filtered dataset
    df_filtered.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_dataset.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    filter_columns(input_file, output_file)
