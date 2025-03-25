import pandas as pd
import os

# File paths
old_file_path = '/Users/zacharylee/TestTest/llm-annotations/datasets/helmet/data_with_corrected_importance_decimals.csv'
new_file_path = '/Users/zacharylee/TestTest/llm-annotations/datasets/helmet/helmet.csv'

# Load the CSV file
df = pd.read_csv(old_file_path)

# Rename the column
df.rename(columns={'normalized_importance': 'confidence_normalized'}, inplace=True)

# Save the updated DataFrame to the new file
df.to_csv(new_file_path, index=False)

# Optionally, remove the old file
os.remove(old_file_path)