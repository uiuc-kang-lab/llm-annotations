import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define LLM-only baseline relative error
llm_only_error = 0.2557251908396947

# Sample Data (Replace with actual results)
sample_sizes = [10, 50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000]
uniform_relative_errors = [0.18, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.007, 0.005, 0.004]
control_variate_relative_errors = [0.15, 0.09, 0.045, 0.028, 0.018, 0.013, 0.009, 0.007, 0.006, 0.0045, 0.0035]
llm_human_relative_errors = [0.17, 0.095, 0.048, 0.029, 0.019, 0.014, 0.01, 0.0075, 0.0065, 0.005, 0.0038]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot LLM-only baseline as a horizontal line
ax.axhline(y=llm_only_error, color='r', linestyle='--', label='LLM Only (Baseline)')

# Plot relative error for different methods
ax.plot(sample_sizes, uniform_relative_errors, marker='o', label='Uniform Sampling')
ax.plot(sample_sizes, control_variate_relative_errors, marker='s', label='Control Variate')
ax.plot(sample_sizes, llm_human_relative_errors, marker='^', label='LLM + Human')

# Labels and title
ax.set_xlabel('Sample Size (# human labels)')
ax.set_ylabel('Relative Error')
ax.set_title('Relative Error Comparison of Different Sampling Methods')
ax.legend()
ax.grid(True)

# Show the plot
plt.show()
