# ============================================================
# Salary Prediction - Advanced Analysis
# SyntecHub - Project 3
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

print("All libraries imported successfully!")

# ============================================================
# STEP 2: Load dataset
# ============================================================
print("\n--- Loading Dataset ---")
df = pd.read_csv('dataset/Salary Prediction of Data Professions.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ============================================================
# STEP 3: Clean and encode
# ============================================================
print("\n--- Cleaning & Encoding ---")
df = df.dropna()

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
drop_cols = ['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE']
categorical_cols = [c for c in categorical_cols if c not in drop_cols]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"Shape after cleaning: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================
# STEP 4: Advanced EDA
# ============================================================
print("\n--- Advanced EDA ---")

# Plot 1: Salary by Designation boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['SALARY'], bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Salary Distribution')
axes[0].set_xlabel('Salary')
axes[0].set_ylabel('Frequency')

axes[1].scatter(df['YEARS EXPERIENCE'], df['SALARY'],
                c=df['DESIGNATION'], cmap='viridis', alpha=0.6)
axes[1].set_title('Experience vs Salary (coloured by Designation)')
axes[1].set_xlabel('Years of Experience')
axes[1].set_ylabel('Salary')
plt.tight_layout()
plt.savefig('eda_overview.png')
plt.show()
print("Saved: eda_overview.png")

# Plot 2: Feature correlation with salary
plt.figure(figsize=(10, 6))
corr = df.corr()['SALARY'].drop('SALARY').sort_values()
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr]
corr.plot(kind='barh', color=colors)
plt.title('Feature Correlation with Salary')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.show()
print("Saved: feature_correlation.png")

# ============================================================
# STEP 5: Prepare data
# ============================================================
X = df.drop('SALARY', axis=1)
y = df['SALARY']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# ============================================================
# STEP 6: Single feature model
# ============================================================
print("\n--- Model 1: Single Feature ---")
X_single_train = X_train[['YEARS EXPERIENCE']]
X_single_test  = X_test[['YEARS EXPERIENCE']]

model_single = LinearRegression()
model_single.fit(X_single_train, y_train)
y_pred_single = model_single.predict(X_single_test)

rmse_single = np.sqrt(mean_squared_error(y_test, y_pred_single))
r2_single   = r2_score(y_test, y_pred_single)
print(f"RMSE: {rmse_single:.2f} | R²: {r2_single:.4f}")

# ============================================================
# STEP 7: Multiple feature model
# ============================================================
print("\n--- Model 2: Multiple Features ---")
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))
r2_multi   = r2_score(y_test, y_pred_multi)
print(f"RMSE: {rmse_multi:.2f} | R²: {r2_multi:.4f}")

# ============================================================
# STEP 8: Detailed comparison plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Single feature plot
axes[0].scatter(y_test, y_pred_single, alpha=0.4, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title(f'Single Feature\nR² = {r2_single:.4f} | RMSE = {rmse_single:.0f}')
axes[0].set_xlabel('Actual Salary')
axes[0].set_ylabel('Predicted Salary')

# Multiple features plot
axes[1].scatter(y_test, y_pred_multi, alpha=0.4, color='#2ecc71')
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_title(f'Multiple Features\nR² = {r2_multi:.4f} | RMSE = {rmse_multi:.0f}')
axes[1].set_xlabel('Actual Salary')
axes[1].set_ylabel('Predicted Salary')

plt.suptitle('Model Comparison — Actual vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison_detailed.png')
plt.show()
print("Saved: model_comparison_detailed.png")

# Plot: R² and RMSE comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['Single Feature', 'Multiple Features']
r2_scores = [r2_single, r2_multi]
rmse_scores = [rmse_single, rmse_multi]

axes[0].bar(models, r2_scores, color=['#5b8db8', '#2ecc71'], width=0.4)
axes[0].set_title('R² Score Comparison')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

axes[1].bar(models, rmse_scores, color=['#5b8db8', '#2ecc71'], width=0.4)
axes[1].set_title('RMSE Comparison (lower = better)')
axes[1].set_ylabel('RMSE')
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + 100, f'{v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.show()
print("Saved: metrics_comparison.png")

# ============================================================
# STEP 9: Model comparison summary
# ============================================================
print("\n--- Final Comparison ---")
print(f"{'Model':<25} {'RMSE':<15} {'R²':<10}")
print("-" * 50)
print(f"{'Single Feature':<25} {rmse_single:<15.2f} {r2_single:<10.4f}")
print(f"{'Multiple Features':<25} {rmse_multi:<15.2f} {r2_multi:<10.4f}")

if r2_multi > r2_single:
    print("\n✅ Multiple Feature model is BETTER!")
    best_model = model_multi
else:
    print("\n✅ Single Feature model is BETTER!")
    best_model = model_single

# ============================================================
# STEP 10: Save best model
# ============================================================
os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/salary_advanced_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Model saved to model/salary_advanced_model.pkl")

# ============================================================
# STEP 11: Example predictions
# ============================================================
print("\n--- Example Predictions ---")
sample = X_test.iloc[:5]
predictions = best_model.predict(sample)
actual = y_test.iloc[:5].values

print(f"\n{'Sample':<10}{'Predicted':<20}{'Actual':<15}")
print("-" * 45)
for i, (pred, act) in enumerate(zip(predictions, actual)):
    print(f"{i+1:<10}{pred:<20.2f}{act:<15.2f}")