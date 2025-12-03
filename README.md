# Sovereign Debt Crisis Prediction

*A Machine Learning Project*
Data & analysis details sourced from: 

---

## 📌 Overview

This project uses supervised and unsupervised machine learning models to predict the likelihood of a **sovereign debt crisis within the next 3 years**. Using macroeconomic indicators from the World Bank and crisis labels from Harvard’s Global Crisis Dataset, the models learn patterns that distinguish stable vs. crisis-prone economies.

---

## 📂 Dataset

* **Sources:** World Bank Open Data + Harvard Global Crisis Data
* **Countries:** 68
* **Years:** 1961–2024
* **Final rows:** 3,526 samples
* **Indicators used:**

  * GDP growth
  * Inflation
  * Domestic credit to private sector
  * Current account balance
  * Total reserves

Each indicator is transformed into **5-year rolling features**:
**mean**, **standard deviation**, **trend**.

---

## 🛠 Preprocessing

* Merged and cleaned 9 economic datasets
* Removed sparse features
* Applied **IterativeImputer** per country
* Engineered 15 total features from 5-year windows
* Normalized all features using StandardScaler

This creates a more realistic representation of economic stability, volatility, and momentum.

---

## 🤖 Models Implemented

### 🔵 Random Forest

* **ROC–AUC:** 0.932
* Strongest overall performance
* Robust to imbalance
* Top features: domestic credit, GDP volatility, reserve trends

### 🟢 Gradient Boosting

* **ROC–AUC:** 0.911
* Better crisis detection than Logistic Regression
* Well-calibrated probability outputs

### 🟡 Logistic Regression

* **ROC–AUC:** 0.865
* Useful baseline but misses many crisis cases due to linearity

### 🔴 K-Means Clustering

* Optimal **K = 3** (silhouette = 0.806)
* Identified a high-risk cluster (~17.8% crisis rate)

### 🟣 Gaussian Mixture Models (GMM)

* Optimal **8–9 components** (AIC/BIC)
* Found multiple high-vulnerability subgroups (>40% crisis probability)

---

## 📊 Key Findings

* **Random Forest** is the most accurate and interpretable model.
* **Gradient Boosting** provides stronger crisis detection and better calibration.
* **Logistic Regression** underperforms for nonlinear macroeconomic dynamics.
* **K-Means & GMM** reveal natural economic clusters aligned with risk levels.
* **Five-year rolling features** significantly improve predictive power.

---

## 🚀 Future Directions

* Extend predictions to 5–10 years ahead
* Incorporate more macro-financial indicators
* Try advanced models (XGBoost, CatBoost, LSTMs)
* Build an interactive early-warning dashboard
* Perform country-specific and regional risk modeling

---

## 📚 References

Full references and detailed analysis are available in the project report. 

Just tell me!
