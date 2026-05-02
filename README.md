# NYC Airbnb Price Prediction

This repository contains our final project for **STATGR 5243**, focused on building an end-to-end machine learning pipeline for **Airbnb price prediction in New York City**.

The project uses the **New York City Airbnb Open Data** dataset from Kaggle and covers the full workflow:

- data cleaning
- exploratory data analysis
- feature engineering
- supervised model development
- model comparison and selection
- an interactive Streamlit dashboard
- a LaTeX report for Overleaf

## Project Structure

```text
.
├── 5243_Project4.ipynb
├── AB_NYC_2019.csv
├── cleaned_data.csv
├── STATGR_5243_Final_Project (1).pdf
├── streamlit_app.py
└── README.md
```

## Dataset

- File: `AB_NYC_2019.csv`
- Source: [New York City Airbnb Open Data on Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/data)

The raw dataset includes listing information such as:

- borough and neighborhood
- latitude and longitude
- room type
- minimum nights
- number of reviews
- reviews per month
- host listing count
- availability
- nightly price

## Repository Files

- `5243_Project4.ipynb`
  - main notebook containing the complete workflow
- `AB_NYC_2019.csv`
  - raw Airbnb dataset
- `cleaned_data.csv`
  - cleaned dataset produced during preprocessing
- `streamlit_app.py`
  - Streamlit dashboard for interactive price prediction
- `STATGR_5243_Final_Project (1).pdf`
  - project report in PDF format
- `README.md`
  - repository overview and usage notes

## Main Notebook

The main analysis is in `5243_Project4.ipynb`.

This notebook includes:

1. data quality assessment
2. cleaning and preprocessing
3. EDA and unsupervised learning
4. feature engineering
5. model tuning and evaluation
6. final model selection
7. model export for deployment

## Modeling Summary

The project compares three supervised learning models:

- Elastic Net
- Random Forest
- Gradient Boosting

The final deployed model is a **Tuned Random Forest**. Although Gradient Boosting performs almost identically, Random Forest was selected as the final deployment model because it provides a strong balance of performance, robustness, interpretability, and dashboard usability.

## Streamlit Dashboard

The repository also includes an interactive Streamlit app:

- `streamlit_app.py`

The dashboard allows users to:

- enter listing characteristics
- generate a predicted nightly Airbnb price
- view the transformed feature inputs used by the model

### Important Note

The Streamlit app expects a saved model file named:

- `final_model.pkl`

If this file is not present yet in your local environment, run the notebook first and execute the model export cell to generate it.

### Run the Streamlit App

Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

Then start the app:

```bash
streamlit run streamlit_app.py
```

By default, Streamlit will open the app in your browser at:

```text
http://localhost:8501
```

## Report

The repository includes the final project report as:

- `STATGR_5243_Final_Project (1).pdf`

## How to Reproduce the Project

1. Download or clone this repository.
2. Make sure `AB_NYC_2019.csv` is in the project root.
3. Open and run `5243_Project4.ipynb`.
4. Run the final model export cell to create `final_model.pkl`.
5. Launch the Streamlit dashboard with `streamlit run streamlit_app.py`.

## Requirements

Recommended Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit

## Notes

- The modeling workflow is based on the cleaned 2019 NYC Airbnb dataset.
- `cleaned_data.csv` is included as a processed version of the dataset used in the workflow.
- Some dashboard inputs use simplified defaults for deployment convenience.
- The final report PDF summarizes the same workflow presented in the notebook.

## Acknowledgment

Dataset credit goes to the Kaggle dataset publisher and the original NYC Airbnb Open Data source.
