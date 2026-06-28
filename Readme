# 💼 Salary Prediction — Advanced Analysis
### SyntecHub Internship — Project 3

An advanced Machine Learning project that predicts employee salaries using **Multiple Linear Regression** with feature scaling, in-depth model comparison, and detailed visualisations.

---

## 📌 Project Overview

This project builds on basic salary prediction by adding **StandardScaler feature scaling**, advanced EDA visualisations, and a detailed side-by-side comparison of single vs multiple feature regression models.

---

## 📂 Project Structure

```
Salary-Prediction-Advanced/
│
├── dataset/
│   └── Salary Prediction of Data Professions.csv
├── model/
│   ├── salary_advanced_model.pkl  # Saved best model
│   └── scaler.pkl                 # Saved scaler
├── salary_prediction_advanced.py  # Main Python script
├── eda_overview.png               # EDA plot
├── feature_correlation.png        # Feature correlation plot
├── model_comparison_detailed.png  # Side by side model comparison
├── metrics_comparison.png         # R² and RMSE comparison
├── requirements.txt               # Required libraries
└── README.md                      # Project documentation
```

---

## 📊 Dataset

- **Source:** Salary Prediction of Data Professions
- **Size:** 500 samples × 9 features (after cleaning)

| Feature | Description |
|---|---|
| YEARS EXPERIENCE | Years of work experience |
| AGE | Employee age |
| DESIGNATION | Job role |
| RATINGS | Performance rating |
| UNIT | Department |
| LEAVES USED | Leaves used |
| LEAVES REMAINING | Leaves remaining |
| SEX | Gender |
| **SALARY** | **Target — Annual salary** |

---

## ⚙️ Tech Stack

- Python 3.14
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sharmis0713-bit/Salary-Prediction-Advanced.git
cd Salary-Prediction-Advanced
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the script**
```bash
python salary_prediction_advanced.py
```

---

## 📈 Results

| Metric | Single Feature | Multiple Features |
|---|---|---|
| RMSE | 21,679.43 | 21,291.54 |
| R² Score | 0.5161 | 0.5332 |

✅ **Multiple Feature model performed better on both metrics!**

---

## 🔍 Key Findings

- **Years of Experience** is the strongest salary predictor (correlation: 0.70)
- **Age** is equally strong (correlation: 0.69) — experience and age naturally increase together
- **Designation** has a moderate positive impact (0.21) — senior roles earn more
- Feature scaling (StandardScaler) ensures all features contribute equally to the model
- Multiple Linear Regression consistently outperforms single feature regression

---

## 📉 Visualisations

| Plot | Description |
|---|---|
| `eda_overview.png` | Salary distribution + Experience vs Salary coloured by Designation |
| `feature_correlation.png` | Horizontal bar chart of each feature's correlation with salary |
| `model_comparison_detailed.png` | Side-by-side actual vs predicted scatter plots for both models |
| `metrics_comparison.png` | R² and RMSE bar chart comparison |

---

## 🙋 Author

**Sharmi S**
B.Sc. Artificial Intelligence & Machine Learning
Hindusthan College of Arts and Science, Coimbatore

---

*Project completed as part of SyntecHub ML Internship*
