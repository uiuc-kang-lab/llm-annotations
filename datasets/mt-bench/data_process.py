import pandas as pd

# Filepath for the CSV file
csv_file = '/mnt/c/Users/krish/Documents/UIUC/Research/Daniel Kang/LLM + Human Annotations/llm-annotations/mt_bench_with_gpt_confidence.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Rename the columns
df.rename(columns={
    'winner_human': 'gold_label',
    'winner_gpt4': 'gpt_label',
    'gpt4_confidence': 'confidence_normalized',
}, inplace=True)

# Normalize the gold_label column to lowercase
# df['gold_label'] = df['gold_label'].str.strip().str.lower()

# Save the updated DataFrame back to the CSV file
df.to_csv('mt-bench.csv', index=False)