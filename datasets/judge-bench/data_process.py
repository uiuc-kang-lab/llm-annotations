import pandas as pd

# Load the CSV file
file_path = '/Users/zacharylee/TestTest/llm-annotations/datasets/judge-bench/judge_bench.csv'
df = pd.read_csv(file_path)

# Rename the columns
df.rename(columns={
    'Query Risk Level (Human)': 'gold_label',
    'Query Risk Level (GPT-4o)': 'gpt_label',
    'importance_weight': 'confidence_normalized'
}, inplace=True)

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)