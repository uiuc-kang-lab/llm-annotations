from openai import OpenAI
import pandas as pd
import os


# Use an environment variable for the API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load the dataset
file_path = "../helmet/data.csv"
df = pd.read_csv(file_path)

def get_severity_score(narrative):
    """Prompt GPT to rate the severity of an incident from 1-100 and print response."""
    prompt = f"""
    You are an expert in injury assessment. Based on the following incident description, rate the severity of the injury on a scale of 1-100, where 1 is minor (e.g., a small scratch) and 100 is life-threatening (e.g., severe head trauma). Respond with only a number and no additional text.
    
    Incident description: "{narrative}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are a medical expert assessing injury severity."},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )

        score = response.choices[0].message.content.strip()
        print(f"GPT Response for Narrative: {narrative}\nSeverity Score: {score}\n")

        return int(score) if score.isdigit() else None

    except Exception as e:
        print(f"Error processing: {e}")
        return None

# Apply the function to all narratives
df["importance_score"] = df["narrative"].apply(get_severity_score)

# Ensure 'importance_score' only contains valid numbers
df = df.dropna(subset=["importance_score"])
df["importance_score"] = df["importance_score"].astype(int)

# Ensure the save directory exists
output_dir = "../helmet"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the updated dataset
output_file = os.path.join(output_dir, "data_with_importance_scores.csv")
df.to_csv(output_file, index=False)

print(f"Processing complete. File saved as {output_file}")
