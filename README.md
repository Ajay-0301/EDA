# Case Study: Exploratory Data Analysis, Statistical Modeling, and Deep Learning for Bank Term Deposit Subscription Prediction

[![Course](https://img.shields.io/badge/Course-U21ADP05%20EDA%20%26%20Visualization-blue.svg)](https://kpriet.ac.in/)
[![Institution](https://img.shields.io/badge/Institution-KPR%20Institute%20of%20Engineering%20%26%20Technology-navy.svg)](https://kpriet.ac.in/)
[![Python](https://img.shields.io/badge/Python-3.13.7-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Assignment%20II-Complete%20%26%20Verified-success.svg)]()

---

## Student Details & Course Metadata
* **Student Name:** AJAY SABARISH A
* **Roll Number:** 24AD007
* **Department:** Department of Artificial Intelligence and Data Science
* **Institution:** KPR Institute of Engineering and Technology, Coimbatore, Tamil Nadu
* **Course Code & Title:** `U21ADP05` — Exploratory Data Analysis and Visualization
* **Assignment:** Assignment II (Case Study)
* **Year / Semester:** III-YR / V Semester
* **Academic Year:** 2026–2027 (Odd Semester)
* **Submission Date:** 15 October 2026
* **GitHub Repository:** [https://github.com/Ajay-0301/EDA](https://github.com/Ajay-0301/EDA)

---

## 1. Problem Statement & Real-World Relevance
In the retail banking industry, outbound telemarketing campaigns represent one of the primary mechanisms for marketing long-term deposit contracts. However, cold telemarketing contacts incur substantial operational costs, customer friction, and low response rates. Identifying customer propensity to subscribe before dialing—conditioned on demographic characteristics, financial background, past campaign interactions, and prevailing macroeconomic indicators—allows banks to allocate telemarketing resources effectively while elevating conversion yields.

This project delivers an end-to-end, rigorous data science investigation featuring:
1. **Data Quality & Schema Ingestion:** Full audit of 41,188 raw records, deduplication (12 duplicates dropped), and categorical sentinel auditing.
2. **Domain-Driven Feature Engineering:** Creation of 6 financial and macroeconomic features to uncover latent interaction patterns.
3. **Probability Distribution Modeling (CO4):** Fitting continuous (Normal, Log-Normal, Gamma) and discrete (Poisson vs Negative Binomial) distributions; testing goodness-of-fit via Kolmogorov-Smirnov (KS) statistics, AIC, Q-Q diagnostics, and formal over-dispersion testing.
4. **Hypothesis Testing:** Pearson's Chi-Square tests of independence ($\chi^2$), Cramer's V effect sizes, Mann-Whitney U tests, and Rank-Biserial correlation effect sizes.
5. **Ben Fry's 7 Stages of Data Visualization:** Comprehensive mapping covering Acquire, Parse, Filter, Mine, Represent, Refine, and Interact (including an interactive Plotly HTML dashboard).
6. **Leakage-Free Partitioning:** Stratified 70% / 15% / 15% train/val/test split with pipeline fitting strictly on training data.
7. **Deep Learning Model Development (Enrichment):** PyTorch Tabular Multilayer Perceptron (MLP) with Batch Normalization, Dropout, and ReLU, tuned across 9 hyperparameter configurations on validation data only.
8. **Threshold Calibration & Benchmarking:** Decision threshold optimization on validation data to address class imbalance, benchmarked against Logistic Regression and Random Forest models.
9. **Permutation Feature Importance:** Identification of primary drivers of model predictions.

---

## 2. Dataset Information
* **Dataset Name:** UCI Bank Marketing Dataset (`bank-additional-full.csv`)
* **Source:** UCI Machine Learning Repository
* **Direct Archive Link:** [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Primary Citation:** S. Moro, P. Cortez, and P. Rita, *"A data-driven approach to predict the success of bank telemarketing,"* *Decision Support Systems*, vol. 62, pp. 22–31, 2014.
* **Volume:** 41,188 raw records (41,176 cleaned records after dropping 12 exact duplicates).
* **Attributes:** 21 raw columns (13 numerical, 7 categorical, 1 binary target `y`).
* **Target Variable:** `y` — Has the client subscribed to a term deposit? (`no`: 36,537 [88.73%], `yes`: 4,639 [11.27%]).
* **Imbalance Ratio:** 7.88 : 1.

---

## 3. Repository Architecture
```
EDA/
├── README.md                                  # Comprehensive project documentation
├── requirements.txt                           # Frozen environment dependencies
├── .gitignore                                 # Git ignore file for Python artifacts
├── data/
│   ├── bank-additional-full.csv               # Raw dataset (41,188 rows)
│   └── bank-additional-names.txt              # Official UCI metadata & attribute dictionary
├── notebooks/
│   └── EDA_Bank_Marketing_Case_Study.ipynb    # Fully executed Jupyter notebook with outputs & charts
└── figures/                                   # High-resolution (150 dpi) chart gallery
    ├── fig01_target_distribution.png
    ├── fig02_age_distribution_and_fit.png
    ├── fig03_duration_lognormal_fit.png
    ├── fig04_discrete_distribution_campaign.png
    ├── fig05_qq_plots_comparison.png
    ├── fig06_job_vs_deposit.png
    ├── fig07_education_marital_rates.png
    ├── fig08_previous_outcome_effect.png
    ├── fig09_correlation_heatmap.png
    ├── fig10_economic_indicators_pairplot.png
    ├── fig11_euribor_vs_duration_scatter.png
    ├── fig12_age_group_and_housing_interaction.png
    ├── fig13_outliers_boxplots.png
    ├── fig14_hyperparameter_search.png
    ├── fig15_training_loss_curves.png
    ├── fig16_training_auc_curves.png
    ├── fig17_confusion_matrix.png
    ├── fig18_roc_and_pr_curves.png
    ├── fig19_prediction_distribution.png
    ├── fig20_permutation_feature_importance.png
    └── interactive_economic_duration_churn.html  # Standalone interactive Plotly HTML
```

---

## 4. Key Empirical Benchmark Results
All metrics below were computed on the **held-out Test partition (6,177 unseen records)**:

| Model / Pipeline | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | True Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.50 | 91.26% | 68.31% | 41.81% | 0.5187 | 0.9401 | 0.6007 | 291 | 405 |
| **Random Forest (Baseline)** | 0.50 | 90.93% | 70.99% | 33.05% | 0.4510 | 0.9453 | 0.6534 | 230 | 466 |
| **PyTorch Tabular MLP (Default)** | 0.50 | 91.47% | 62.23% | 61.78% | 0.6200 | **0.9481** | **0.6579** | 430 | 266 |
| **PyTorch Tabular MLP (Calibrated)** | **0.44** | 91.32% | 59.93% | **69.40%** | **0.6431** | **0.9481** | **0.6579** | **483** | **213** |

### Critical Takeaways:
1. **Recall Uplift:** Under the validation-calibrated decision threshold (0.44), the PyTorch Tabular MLP identified **483 out of 696 actual subscribers (69.40% recall)**, compared to only 230 for Random Forest (33.05%) and 291 for Logistic Regression (41.81%).
2. **Discrimination Superiority:** The PyTorch MLP achieved the highest overall **ROC-AUC (0.9481)** and **PR-AUC (0.6579)**.
3. **Distribution Modeling:** Call duration is strictly non-normal and conforms exceptionally well to a **Log-Normal distribution** ($KS = 0.0163$, $AIC = 534,482$ vs Normal $KS = 0.1695$, $AIC = 574,513$).
4. **Over-Dispersion:** Outreach campaign contacts exhibit a variance-to-mean ratio of **2.99** (Mean = 2.57, Variance = 7.67), proving substantial over-dispersion and invalidating standard Poisson models in favor of the Negative Binomial.
5. **Permutation Importance:** Call duration is the primary driver of prediction (mean validation AUC drop of 0.2086), followed by the employment variation rate (0.0538) and contact month (0.0162).

---

## 5. Ben Fry's 7 Stages of Data Visualization
| Stage | Definition | Project Implementation |
| :--- | :--- | :--- |
| **1. Acquire** | Obtain the raw data | Direct HTTP download from the UCI Machine Learning Repository archive. |
| **2. Parse** | Structure and parse format | Loaded semicolon-separated raw records into tabular Pandas DataFrame; schema typing. |
| **3. Filter** | Remove noise / unneeded data | Dropped 12 duplicate records; isolated 999 pdays sentinels; audited categorical 'unknowns'. |
| **4. Mine** | Apply statistical & ML methods | Parametric distribution fitting (Normal, Log-normal, Gamma), KS tests, Chi-Square, Mann-Whitney U. |
| **5. Represent** | Choose primary visual models | Histograms, density curves, Q-Q plots, grouped rate charts, correlation heatmaps, ROC/PR curves. |
| **6. Refine** | Polish design and clarity | Applied 150-dpi rendering, high-contrast palettes, annotations, and units. |
| **7. Interact** | Dynamic data exploration | Deployed interactive Plotly scatter/bubble chart with rich multi-variable hover tooltips. |

---

## 6. How to Run and Reproduce
### Prerequisites
* Windows, Linux, or macOS with **Python 3.10+** (tested on Python 3.13.7)
* Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ajay-0301/EDA.git
cd EDA
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Executable Jupyter Notebook
To run the fully commented notebook programmatically and regenerate all outputs:
```bash
jupyter nbconvert --to notebook --execute notebooks/EDA_Bank_Marketing_Case_Study.ipynb --output notebooks/EDA_Bank_Marketing_Case_Study.ipynb
```
Or open interactively in Jupyter Lab:
```bash
jupyter lab notebooks/EDA_Bank_Marketing_Case_Study.ipynb
```

### Step 4: View Figures and Interactive Visualizations
* All 20 publication figures are located in `figures/`.
* The interactive Plotly chart can be viewed by opening `figures/interactive_economic_duration_churn.html` in any web browser.

---

## 7. Academic References
1. **S. Moro, P. Cortez, and P. Rita**, *"A data-driven approach to predict the success of bank telemarketing,"* *Decision Support Systems*, vol. 62, pp. 22–31, 2014. DOI: [10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
2. **B. Fry**, *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. Sebastopol, CA: O'Reilly Media, 2008. ISBN: 978-0596514556.
3. **I. Goodfellow, Y. Bengio, and A. Courville**, *Deep Learning*. Cambridge, MA: MIT Press, 2016.
4. **F. Pedregosa et al.**, *"Scikit-learn: Machine learning in Python,"* *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
5. **A. Paszke et al.**, *"PyTorch: An imperative style, high-performance deep learning library,"* in *Advances in Neural Information Processing Systems (NeurIPS 32)*, 2019, pp. 8024–8035.
6. **T. Fawcett**, *"An introduction to ROC analysis,"* *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006. DOI: [10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010).

---
*Created as part of the academic curriculum for Course U21ADP05 at KPR Institute of Engineering and Technology.*
