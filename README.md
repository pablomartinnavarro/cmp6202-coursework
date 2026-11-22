# CMP6202 — Predicting English Premier League Match Outcomes Using Pre-Match Tactical Metrics and Supervised Machine Learning

This repository contains the reproducible codebase, processed dataset, and visualisations for the **CMP6202 Artificial Intelligence and Machine Learning** individual coursework (Birmingham City University, 2026–2027).

## Project Overview
The project formulates a leakage-free binary classification pipeline to predict English Premier League home victories (`HomeWin = 1`) versus non-home outcomes (`HomeWin = 0`, draws and away wins) using strictly pre-match indicators across 11 seasons (2014–2024, **N = 4,178 fixtures**). Three supervised classifier families—**Logistic Regression**, **Decision Tree**, and **Random Forest**—are benchmarked against a majority-class **DummyClassifier** baseline using 5-fold stratified cross-validation, domain-driven feature engineering, `GridSearchCV` hyperparameter tuning, and a locked 20% unseen holdout test set (**N = 836 fixtures**).

## Repository Structure
```text
cmp6202-coursework/
├── data/
│   └── processed/
│       └── premier_league_2014_2024.csv   # Filtered EPL 2014–2024 dataset (4,178 matches)
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis (Figures 1–3)
│   ├── 02_modelling.ipynb                 # Train/test split, default models & seen-data audit (Figure 4)
│   └── 03_evaluation_and_tuning.ipynb     # 5-Fold CV, feature engineering, GridSearchCV & test evaluation (Figures 5–7)
├── report/
│   └── figures/                           # Exported high-resolution visualisations (PNG, 300 dpi)
└── README.md
```

## Dataset Source & Licence
- **Source Dataset:** *Club Football Match Data* by Adam Gábor (integrating match records from [Football-Data.co.uk](https://www.football-data.co.uk/) and dynamic club ratings from [ClubElo](http://clubelo.com/)).
- **Raw Data Access:** The full raw repository (238,858 global matches, ~57 MB) is publicly accessible via [Kaggle](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) and [GitHub](https://github.com/xgabora/Club-Football-Match-Data).
- **Processed Subset Included:** `data/processed/premier_league_2014_2024.csv` contains the filtered English Premier League (`E0`) cohort from 2014 to 2024 inclusive (4,178 rows), included directly in this repository for immediate reproducibility.
- **Licence:** Distributed under the **MIT License**, permitting unrestricted academic and research use.

## Environment Setup & How to Run
### 1. Prerequisites
- Python **3.10+** (developed and tested on Python **3.13.5**)
- Jupyter Notebook / VS Code Jupyter extension

### 2. Clone the Repository and Install Dependencies
```bash
git clone [https://github.com/pablomartinnavarro/cmp6202-coursework.git](https://github.com/pablomartinnavarro/cmp6202-coursework.git)
cd cmp6202-coursework
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Execute the Notebooks in Order
Open the repository in VS Code or launch Jupyter (`jupyter notebook`) and run all cells sequentially in the following order:
1. `notebooks/01_eda.ipynb` — Inspects class balance, Elo differentials, and pairwise correlations; exports Figures 1–3 to `report/figures/`.
2. `notebooks/02_modelling.ipynb` — Applies stratified 80/20 splitting (`random_state=42`), training-only median imputation and scaling, default model fitting, and seen-data overfitting evaluation; exports Figure 4.
3. `notebooks/03_evaluation_and_tuning.ipynb` — Runs leakage-free 5-fold stratified cross-validation pipelines, engineers `FormDiff5` and margin-normalised implied probabilities, executes ROC-AUC `GridSearchCV` tuning, evaluates final models on the unseen test set (`N = 836`), and exports Figures 5–7.

## Summary of Final Holdout Test Results (N = 836)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Dummy) | 0.550 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Logistic Regression (Tuned)** | **0.672** | 0.629 | **0.660** | **0.644** | **0.718** |
| Decision Tree (Tuned) | 0.661 | 0.644 | 0.553 | 0.595 | 0.697 |
| Random Forest (Tuned) | 0.670 | **0.672** | 0.519 | 0.586 | 0.716 |