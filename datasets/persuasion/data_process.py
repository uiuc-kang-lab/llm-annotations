import pandas as pd

# Filepath for the CSV file
csv_file = '/Users/zacharylee/TestTest/llm-annotations/datasets/persuasion/persuasion.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Rename the columns
df.rename(columns={
    'Gold Label': 'gold_label',
    'GPT Label': 'gpt_label',
    'Normalized Confidence': 'confidence_normalized'
}, inplace=True)

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_file, index=False)