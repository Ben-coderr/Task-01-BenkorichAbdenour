#  Data Science Project 1: Advanced EDA & Feature Engineering
**DecodeLabs Internship | Batch 2026**

##  Overview
This project transforms the raw Titanic dataset into a highly structured, machine-learning-ready format. It strictly adheres to an enterprise-grade **Input → Process → Output (IPO) Architecture** and employs runtime data validation.

##  Architecture & Features
* **IPO Pattern:** Clear separation of data ingestion, transformation logic, and final validation.
* **Intelligent Imputation:** Uses `KNNImputer` for numerical gaps (e.g., Age) and statistical modes for categorical gaps.
* **Advanced Feature Engineering:** Extracts social titles from names, constructs family size indicators, and groups continuous variables into strategic bins.
* **Data Integrity:** Employs **Pandera** for rigorous runtime schema validation. Output data strictly conforms to defined contracts (no missing values, correct datatypes).
* **Correlation Sweeps:** Automated multi-collinearity detection (Pearson correlation > 0.80) to ensure feature independence.

##  Repository Structure
* `project1_solution.ipynb`: The main professional Jupyter Notebook containing rich Markdown steps, inline visualizations, and documented execution cells.
* `project1_solution.py`: The pure Python equivalent of the pipeline.
* `titanic_cleaned_engineered.csv`: The final, mathematically validated ML-ready dataset.

##  Getting Started
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib pandera jupyter

# 2. Run the Notebook
jupyter notebook project1_solution.ipynb

# 3. Or run the CLI script
python project1_solution.py
```

##  Output
The pipeline produces robust graphical summaries of feature distributions, missingness matrices, and correlation heatmaps to guarantee the integrity of the predictive features.

---
*Authored with strict adherence to DecodeLabs architectural guidelines.*
