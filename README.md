# LLM Annotations Framework

This repository provides tools for evaluating and comparing various sampling methods (e.g., uniform sampling, importance sampling, control variate) for datasets annotated by LLMs and humans.

---

## 1. Dataset Formatting

Each dataset must be a CSV file containing the following required columns:

| Column Name             | Description                                                                    |
|-------------------------|--------------------------------------------------------------------------------|
| `data_entry`            | The text or data entry being evaluated.                                        |
| `gold_label`            | The ground truth label for the entry (provided by human annotators).           |
| `gpt_label`             | The label predicted by the LLM.                                                |
| `confidence_normalized` | The normalized confidence score of the LLM prediction (range: 0 to 1).         |

### Example Dataset
```csv
data_entry,gold_label,gpt_label,confidence_normalized
"The earth is warming.",1,1,0.95
"Climate change is a hoax.",0,0,0.80
"Renewable energy is the future.",1,1,0.90
```

#### Dataset File Locations
• Place datasets in the datasets directory under respective subfolders (e.g., global_warming, helmet).  
• Ensure the dataset file is named appropriately (e.g., global_warming.csv, helmet.csv).

---

## 2. Setting Up load_dataset

The file load_dataset.py handles loading datasets and computing initial statistics. To add a new dataset:
1. Open load_dataset.py.  
2. Add a new condition:
```python
elif dataset == "new_dataset_name":
    data = pd.read_csv("../llm-annotations/datasets/new_dataset_name/new_dataset_name.csv")
```
3. Ensure the new dataset file is properly formatted as described above.

---

## 3. Setting Up compute_statistics

The compute_statistics function in statistic.py computes dataset-specific statistics. To add a new dataset:
1. Open statistic.py.  
2. Add a new if block:
```python
if dataset == "new_dataset_name":
    # Add logic to compute statistics for the dataset
    return some_statistic
```
3. Ensure the logic aligns with the dataset’s structure and evaluation needs.

---

## 4. Running Tests

### Available Sampling Methods

• **Uniform Sampling** – Replaces a portion of LLM predictions with human labels.  
• **Importance Sampling** – Uses LLM confidence scores to guide sampling.  
• **Control Variate** – Adjusts estimates using LLM confidence as a proxy variable.  
• **LLM Only** – Evaluates the LLM predictions without human labels.

### Commands to Run Tests

Use the following commands in your terminal:

#### Uniform Sampling
```bash
python methods/uniform_sampling.py --dataset <dataset_name> --step_size 100 --repeat 1000 --save_dir results/uniform_sampling/
```

#### Importance Sampling
```bash
python methods/importance_sampling.py --dataset <dataset_name> --max_human_budget 999999 --step_size 100 --repeat 1000 --save_dir results/importance_sampling/
```

#### Control Variate
```bash
python methods/control_variate.py --dataset <dataset_name> --step_size 100 --max_human_budget 999999 --repeat 1000 --save_dir results/control_variate/
```

#### LLM Only
```bash
python methods/llm_only.py <dataset_name>
```

Example for uniform sampling on the global_warming dataset:
```bash
python methods/uniform_sampling.py --dataset global_warming --step_size 100 --repeat 1000 --save_dir results/uniform_sampling/
```

---

## 5. Results

Results are saved as CSV files in the results directory under method-specific subfolders:
• uniform_sampling  
• importance_sampling  
• control_variate  
• llm_human  

Each result CSV file typically contains:
• Dataset – The name of the dataset  
• Human Samples – The number of human-labeled samples used  
• Relative Error – The relative error of the estimate  
• LLM Samples (for uniform sampling) – The number of LLM-labeled samples used  

---

## 6. Plotting Results

To visualize outcomes, use the notebooks in the plots directory. For example:
• plot_10_22.ipynb – Generates comparison plots for sampling methods.  

Ensure the result CSV files are in the correct locations before running the notebook.

---

## 7. Debugging Common Issues

### KeyError in Sampling
• Confirm that the dataset size is not exceeded by max_human_budget.  
• Cap sample_sizes to the dataset size in the sampling scripts.

### Confidence Scores
• Check calibration and meaningfulness of confidence scores.  
• Use balanced or stratified sampling if the dataset is skewed.

---

## 8. Adding a New Sampling Method

1. Create a new script in methods (e.g., new_sampling.py).  
2. Implement the sampling logic, following the existing structure.  
3. Save results (CSV) in the results directory, with an appropriate subfolder.

---

## 9. Directory Structure

```
llm-annotations/
├── datasets/
│   ├── global_warming/
│   ├── helmet/
│   ├── implicit_hate/
│   └── ...
├── methods/
│   ├── uniform_sampling.py
│   ├── importance_sampling.py
│   ├── control_variate.py
│   ├── llm_only.py
│   └── ...
├── results/
│   ├── uniform_sampling/
│   ├── importance_sampling/
│   ├── control_variate/
│   └── ...
├── plots/
│   ├── plot_10_22.ipynb
│   └── ...
└── README.md
```