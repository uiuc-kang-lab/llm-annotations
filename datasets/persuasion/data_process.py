import pandas as pd

# Filepath for the CSV file
csv_file = '/Users/zacharylee/llm-annotations/datasets/persuasion/persuasion.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Rename the columns
df.rename(columns={
    'Gold Label': 'gold_label',
    'GPT Label': 'gpt_label',
    'Normalized Confidence': 'confidence_normalized'
}, inplace=True)

# Normalize the gold_label column to lowercase
df['gold_label'] = df['gold_label'].str.strip().str.lower()

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_file, index=False)