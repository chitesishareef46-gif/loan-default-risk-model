"""
Loan Default Risk Prediction Model
====================================
Author  : Shareef Chitesi
Degree  : BSc Honours Applied Mathematics & Computational Science
          Midlands State University — Year 2
Project : Logistic regression risk model predicting loan default
          probability, built from first principles using NumPy.

Mathematical foundation:
  - Logistic function:  p = 1 / (1 + e^(-z)),  z = Xβ
  - Maximum Likelihood Estimation via Gradient Descent
  - Cross-entropy loss:  L = -Σ[y·log(p) + (1-y)·log(1-p)]
  - Confusion matrix, ROC curve, AUC — model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

BLUE   = "#1B4F72"
RED    = "#C0392B"
ORANGE = "#E67E22"
GREEN  = "#1E8449"
GREY   = "#808B96"
PURPLE = "#6C3483"
LIGHT  = "#D6EAF8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

print("=" * 65)
print("  LOAN DEFAULT RISK PREDICTION MODEL")
print("  Author: Shareef Chitesi | MSU Applied Mathematics Year 2")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC LOAN DATASET (realistic distributions, reproducible)
#     In production this would be a bank's real loan book
# ─────────────────────────────────────────────────────────────────────────────
n = 1000

age            = np.clip(np.random.normal(38, 11, n), 21, 70).astype(int)
income         = np.clip(np.random.lognormal(8.4, 0.5, n), 3000, 80000)
loan_amount    = np.clip(np.random.lognormal(8.0, 0.6, n), 500, 50000)
credit_score   = np.clip(np.random.normal(620, 90, n), 300, 850).astype(int)
employment_yrs = np.clip(np.random.exponential(5, n), 0, 35)
debt_to_income = np.clip(np.random.normal(0.35, 0.15, n), 0.02, 0.95)
num_late_pay   = np.random.poisson(1.2, n)
loan_term      = np.random.choice([12, 24, 36, 48, 60], n, p=[0.15,0.25,0.3,0.2,0.1])

# True underlying risk model (the "ground truth" we're trying to recover)
# Higher debt-to-income, lower credit score, more late payments → higher risk
z_true = (
    1.0
    + 4.5  * debt_to_income
    - 0.008 * credit_score
    + 0.35 * num_late_pay
    - 0.00002 * income
    + 0.00003 * loan_amount
    - 0.05 * employment_yrs
    + np.random.normal(0, 0.8, n)   # noise
)
prob_default = 1 / (1 + np.exp(-z_true))
default = (np.random.rand(n) < prob_default).astype(int)

df = pd.DataFrame({
    "age": age, "income": income.round(0), "loan_amount": loan_amount.round(0),
    "credit_score": credit_score, "employment_years": employment_yrs.round(1),
    "debt_to_income": debt_to_income.round(3), "num_late_payments": num_late_pay,
    "loan_term_months": loan_term, "default": default
})

print(f"\n  Dataset size        : {n} loan applicants")
print(f"  Default rate        : {df.default.mean()*100:.1f}%")
print(f"  Mean credit score   : {df.credit_score.mean():.0f}")
print(f"  Mean debt-to-income : {df.debt_to_income.mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
shuffle_idx = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = shuffle_idx[:split], shuffle_idx[split:]

features = ["age","income","loan_amount","credit_score",
            "employment_years","debt_to_income","num_late_payments","loan_term_months"]

X_raw = df[features].values
y = df["default"].values

# Standardise features (zero mean, unit variance) — required for gradient descent
X_mean = X_raw[train_idx].mean(axis=0)
X_std  = X_raw[train_idx].std(axis=0)
X_scaled = (X_raw - X_mean) / X_std

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Add intercept column
X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
X_test_b  = np.column_stack([np.ones(len(X_test)), X_test])

print(f"\n  Training set        : {len(X_train)} loans")
print(f"  Test set            : {len(X_test)} loans")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOGISTIC REGRESSION FROM SCRATCH (Gradient Descent / MLE)
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_logistic_regression(X, y, lr=0.1, epochs=2000, l2=0.01):
    """
    Logistic regression via batch gradient descent.
    Minimises cross-entropy loss with L2 regularisation.
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    loss_history = []

    for epoch in range(epochs):
        z = X @ beta
        p = sigmoid(z)

        # Cross-entropy loss + L2 penalty
        eps = 1e-10
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += (l2 / (2 * n_samples)) * np.sum(beta[1:] ** 2)
        loss_history.append(loss)

        # Gradient of cross-entropy w.r.t. beta
        gradient = (X.T @ (p - y)) / n_samples
        gradient[1:] += (l2 / n_samples) * beta[1:]   # don't regularise intercept

        beta -= lr * gradient

    return beta, loss_history

print(f"\n── Training Logistic Regression (Gradient Descent) ────────")
beta, loss_history = train_logistic_regression(X_train_b, y_train, lr=0.3, epochs=3000, l2=0.5)
print(f"  Final training loss : {loss_history[-1]:.4f}")
print(f"  Converged after     : {len(loss_history)} iterations")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PREDICTIONS & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
p_train = sigmoid(X_train_b @ beta)
p_test  = sigmoid(X_test_b @ beta)

y_pred_test = (p_test >= 0.5).astype(int)

# Confusion matrix components
tp = np.sum((y_pred_test == 1) & (y_test == 1))
tn = np.sum((y_pred_test == 0) & (y_test == 0))
fp = np.sum((y_pred_test == 1) & (y_test == 0))
fn = np.sum((y_pred_test == 0) & (y_test == 1))

accuracy  = (tp + tn) / len(y_test)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n── Model Performance (Test Set) ─────────────────────────────")
print(f"  Accuracy  : {accuracy*100:.1f}%")
print(f"  Precision : {precision*100:.1f}%  (of predicted defaults, % correct)")
print(f"  Recall    : {recall*100:.1f}%  (of actual defaults, % caught)")
print(f"  F1 Score  : {f1:.3f}")
print(f"\n  Confusion Matrix:")
print(f"                   Predicted: No Default   Predicted: Default")
print(f"  Actual: No Default      {tn:>6}              {fp:>6}")
print(f"  Actual: Default          {fn:>6}              {tp:>6}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ROC CURVE & AUC (manual implementation)
# ─────────────────────────────────────────────────────────────────────────────
thresholds = np.linspace(0, 1, 100)
tpr_list, fpr_list = [], []

for thresh in thresholds:
    pred = (p_test >= thresh).astype(int)
    tp_t = np.sum((pred == 1) & (y_test == 1))
    fp_t = np.sum((pred == 1) & (y_test == 0))
    fn_t = np.sum((pred == 0) & (y_test == 1))
    tn_t = np.sum((pred == 0) & (y_test == 0))

    tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# AUC via trapezoidal rule
fpr_arr = np.array(fpr_list)[::-1]
tpr_arr = np.array(tpr_list)[::-1]
auc = np.trapezoid(tpr_arr, fpr_arr)

print(f"\n  AUC (Area Under ROC Curve): {auc:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE IMPORTANCE (standardised coefficients)
# ─────────────────────────────────────────────────────────────────────────────
feature_importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": beta[1:],
    "Abs_Coefficient": np.abs(beta[1:])
}).sort_values("Abs_Coefficient", ascending=False)

print(f"\n── Feature Importance (Standardised Coefficients) ──────────")
for _, row in feature_importance.iterrows():
    direction = "↑ increases risk" if row["Coefficient"] > 0 else "↓ decreases risk"
    print(f"  {row['Feature']:<20} {row['Coefficient']:>8.3f}   {direction}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  RISK SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def risk_band(p):
    if p < 0.15: return "Low Risk"
    elif p < 0.35: return "Medium Risk"
    elif p < 0.60: return "High Risk"
    else: return "Very High Risk"

df_test = df.iloc[test_idx].copy()
df_test["predicted_prob"] = p_test
df_test["risk_band"] = df_test["predicted_prob"].apply(risk_band)

risk_summary = df_test.groupby("risk_band").agg(
    count=("default", "count"),
    actual_default_rate=("default", "mean"),
    avg_predicted_prob=("predicted_prob", "mean")
).reindex(["Low Risk","Medium Risk","High Risk","Very High Risk"])

print(f"\n── Risk Band Segmentation ───────────────────────────────────")
print(risk_summary.to_string())

# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════

# ── PLOT 1: Training Loss Curve ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(loss_history, color=BLUE, linewidth=2)
ax.set_title("Logistic Regression — Training Loss (Gradient Descent)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/01_training_loss.png", bbox_inches="tight")
plt.close()
print("\n  ✓ Chart 1 saved: Training Loss Curve")

# ── PLOT 2: ROC Curve ──
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr_list, tpr_list, color=PURPLE, linewidth=2.5, label=f"ROC curve (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.5, label="Random classifier (AUC = 0.5)")
ax.fill_between(fpr_list, tpr_list, alpha=0.1, color=PURPLE)
ax.set_title("ROC Curve — Loan Default Prediction", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/02_roc_curve.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 2 saved: ROC Curve")

# ── PLOT 3: Confusion Matrix Heatmap ──
fig, ax = plt.subplots(figsize=(7, 6))
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            annot_kws={"size": 16, "weight": "bold"}, ax=ax)
ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/03_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 3 saved: Confusion Matrix")

# ── PLOT 4: Feature Importance ──
fig, ax = plt.subplots(figsize=(10, 6))
colors_fi = [RED if c > 0 else GREEN for c in feature_importance["Coefficient"]]
ax.barh(feature_importance["Feature"], feature_importance["Coefficient"],
        color=colors_fi, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Importance — Standardised Logistic Regression Coefficients",
             fontsize=12, fontweight="bold", pad=15)
ax.set_xlabel("Coefficient (standardised)", fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=RED, label="Increases default risk"),
                   Patch(facecolor=GREEN, label="Decreases default risk")],
          fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/04_feature_importance.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 4 saved: Feature Importance")

# ── PLOT 5: Risk Band Distribution ──
fig, ax1 = plt.subplots(figsize=(10, 6))
order = ["Low Risk","Medium Risk","High Risk","Very High Risk"]
colors_band = [GREEN, ORANGE, "#D35400", RED]

bars = ax1.bar(order, risk_summary["count"], color=colors_band, alpha=0.8, width=0.55)
ax1.set_ylabel("Number of Applicants", fontsize=11)
ax1.set_title("Risk Band Segmentation — Count vs Actual Default Rate",
              fontsize=12, fontweight="bold", pad=15)

ax2 = ax1.twinx()
ax2.plot(order, risk_summary["actual_default_rate"] * 100, color=BLUE,
         marker="o", markersize=10, linewidth=2.5, label="Actual default rate")
ax2.set_ylabel("Actual Default Rate (%)", fontsize=11, color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)

for i, (cnt, rate) in enumerate(zip(risk_summary["count"], risk_summary["actual_default_rate"])):
    ax1.text(i, cnt + 5, f"n={cnt}", ha="center", fontsize=9)
    ax2.text(i, rate*100 + 3, f"{rate*100:.0f}%", ha="center", fontsize=9,
             color=BLUE, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/05_risk_segmentation.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 5 saved: Risk Band Segmentation")

# ── PLOT 6: Predicted Probability Distribution by Actual Outcome ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(p_test[y_test == 0], bins=25, alpha=0.6, color=GREEN, label="Actual: No Default", density=True)
ax.hist(p_test[y_test == 1], bins=25, alpha=0.6, color=RED, label="Actual: Default", density=True)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision threshold (0.5)")
ax.set_title("Predicted Default Probability Distribution by Actual Outcome",
             fontsize=12, fontweight="bold", pad=15)
ax.set_xlabel("Predicted Probability of Default", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/06_probability_distribution.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 6 saved: Probability Distribution")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPORT TO EXCEL (Power BI ready)
# ─────────────────────────────────────────────────────────────────────────────
with pd.ExcelWriter("outputs/loan_risk_data.xlsx", engine="openpyxl") as writer:
    df_test[features + ["default","predicted_prob","risk_band"]].to_excel(
        writer, sheet_name="Test Predictions", index=False)

    feature_importance.to_excel(writer, sheet_name="Feature Importance", index=False)

    risk_summary.reset_index().to_excel(writer, sheet_name="Risk Segmentation", index=False)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy","Precision","Recall","F1 Score","AUC",
                   "True Positives","True Negatives","False Positives","False Negatives"],
        "Value": [f"{accuracy*100:.1f}%", f"{precision*100:.1f}%", f"{recall*100:.1f}%",
                  f"{f1:.3f}", f"{auc:.3f}", tp, tn, fp, fn]
    })
    metrics_df.to_excel(writer, sheet_name="Model Metrics", index=False)

print("\n  ✓ Excel exported: loan_risk_data.xlsx")
print(f"""
{'='*65}
  SUMMARY
{'='*65}
  Model           : Logistic Regression (from scratch, gradient descent)
  Accuracy        : {accuracy*100:.1f}%
  AUC             : {auc:.3f}
  Top risk factor : {feature_importance.iloc[0]['Feature']}
  Charts saved to outputs/ (6 charts)
{'='*65}
""")
