import pandas as pd

# Filepath for the CSV file
file_path = '/Users/zacharylee/llm-annotations/datasets/implicit_hate/implicit_hate.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Rename the columns
df.rename(columns={'Gold Label': 'gold_label', 'GPT Label': 'gpt_label', 'Normalized Confidence' : 'confidence_normalized'}, inplace=True)

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)