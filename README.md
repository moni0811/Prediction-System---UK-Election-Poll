## 🎯 Purpose of the Analysis

The purpose of this analysis is to:

- Analyze voter behavior based on survey data collected before the recent election
- Build predictive models to classify voter preference between major political parties
- Evaluate and compare models using standard classification metrics
- Identify the most accurate and reliable model for use in an exit poll prediction system

---

## 📂 What's in the Repo

- `UK__Election_Poll.ipynb` – Python script containing the full modeling and evaluation pipeline  
- `README.md` – This file

> **Note:** The dataset (`Election_Data.xlsx`) contains confidential survey responses and is **not included** in this repository. For internal use, ensure the dataset is placed in the project root directory before executing the script.

---

## 🛠 Methods Used

The project workflow includes the following stages:

1. **Data Ingestion & Preliminary Checks**
   - Loaded Excel dataset
   - Generated descriptive statistics and null value analysis
   - Dropped irrelevant columns

2. **Exploratory Data Analysis**
   - Univariate visualizations: histograms, boxplots, countplots
   - Bivariate visualizations: pair plots for interaction insights
   - Checked and interpreted outliers

3. **Data Preparation**
   - Encoded categorical features (`gender`, `vote`)
   - Split the data into training (70%) and testing (30%) subsets

4. **Modeling**
   - Applied three classification algorithms:
     - Logistic Regression
     - Decision Tree
     - Random Forest
   - Trained each model on the training set

5. **Evaluation**
   - Assessed performance on both training and test sets using:
     - Accuracy
     - Confusion Matrix
     - ROC Curves
     - ROC AUC Score
   - Visualized and compared ROC curves across models

---

## 🔍 Results Summary

The models successfully predicted voter preferences with high accuracy. Key takeaways include:

- Logistic Regression, Decision Tree, and Random Forest were all effective
- ROC AUC scores and confusion matrices revealed model strengths and weaknesses
- The **Random Forest** model showed superior balance between training and test accuracy, making it a strong candidate for use in an exit poll system

Final model selection was based on interpretability, consistency, and generalization performance across unseen data.
