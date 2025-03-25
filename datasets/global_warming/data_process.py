import pandas as pd
# No need for MinMaxScaler as we normalize by sum

# Load the CSV file
file_path = '/Users/zacharylee/TestTest/llm-annotations/datasets/global_warming/global_warming.csv'
df = pd.read_csv(file_path)

# Normalize the gpt_confidence column
# Normalize the gpt_confidence column so it sums to 1
df['normalized_gpt_confidence'] = df['gpt_confidence'] / df['gpt_confidence'].sum()
# Rename the column 'normalized_gpt_confidence' to 'confidence_normalized'
df.rename(columns={'normalized_gpt_confidence': 'confidence_normalized'}, inplace=True)
# Save the updated DataFrame back to the CSV
# Drop the duplicate 'confidence_normalized' column if it exists
df = df.loc[:, ~df.columns.duplicated()]

# Save the updated DataFrame back to the CSV
df.to_csv(file_path, index=False)