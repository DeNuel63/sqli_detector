"""
SQL Injection Detection — Model Evaluation
===========================================
Input  : models/model.pkl
         data/processed/features_test.npz / labels_test.npy
         data/processed/feature_names.json
Output : reports/figures/confusion_matrix.png
         reports/figures/roc_curve.png
         reports/figures/precision_recall_curve.png
         reports/figures/shap_summary.png
         reports/figures/shap_top_features.png
         reports/figures/error_analysis.png
         reports/test_metrics.json

Run:
    python src/evaluate.py

⚠  CLASS IMBALANCE REMINDER (1.3:1 SQLi:Benign)
    The test set mirrors the training distribution (57% SQLi / 43% Benign).
    Real production traffic will be far more benign-heavy (99%+ benign).
    When deploying, revisit the classification threshold — the default 0.5
    cutoff may generate too many false positives in production.
    Consider lowering it (e.g. 0.3) to reduce false alarm fatigue.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from scipy.sparse import load_npz
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, recall_score, precision_score,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# ── Paths (aligned with agreed project structure) ─────────────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load model, test features, and feature names
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING MODEL & TEST DATA")
print("=" * 60)

model  = pickle.load(open(os.path.join(MODELS_DIR, "model.pkl"), "rb"))
X_test = load_npz(os.path.join(PROCESSED_DIR, "features_test.npz"))
y_test = np.load(os.path.join(PROCESSED_DIR, "labels_test.npy"))

with open(os.path.join(PROCESSED_DIR, "feature_names.json")) as f:
    feature_names = json.load(f)

# Load raw queries for error analysis
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

print(f"Model          : {type(model).__name__}")
print(f"Test set shape : {X_test.shape}")
print(f"Test Benign    : {(y_test==0).sum():,}  |  Test SQLi: {(y_test==1).sum():,}")
print(f"Feature names  : {len(feature_names):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Generate predictions
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # probability of SQLi

print(f"Predictions generated for {len(y_pred):,} samples.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Core metrics
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — CORE METRICS (HELD-OUT TEST SET)")
print("=" * 60)

f1   = f1_score(y_test, y_pred, average="macro")
auc  = roc_auc_score(y_test, y_prob)
ap   = average_precision_score(y_test, y_prob)
rec0 = recall_score(y_test, y_pred, pos_label=0)
rec1 = recall_score(y_test, y_pred, pos_label=1)
pre0 = precision_score(y_test, y_pred, pos_label=0)
pre1 = precision_score(y_test, y_pred, pos_label=1)

print(classification_report(y_test, y_pred, target_names=["Benign", "SQLi"]))
print(f"  ROC-AUC              : {auc:.4f}")
print(f"  Average Precision    : {ap:.4f}")
print(f"  F1-macro             : {f1:.4f}")

# ⚠  Class imbalance check
print()
if rec0 < 0.85:
    print(f"  ⚠  Recall (benign)  : {rec0:.4f}  BELOW 0.85")
    print("     In production (99%+ benign traffic) this will cause false alarm fatigue.")
    print("     Consider: lowering decision threshold OR enabling SMOTE in train.py.")
else:
    print(f"  ✓  Recall (benign)  : {rec0:.4f}  — within acceptable range")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n  Confusion matrix breakdown:")
print(f"    True  Negatives (benign correctly blocked) : {tn:,}")
print(f"    False Positives (benign flagged as attack) : {fp:,}  ← false alarm rate")
print(f"    False Negatives (attacks that slipped by)  : {fn:,}  ← miss rate")
print(f"    True  Positives (attacks correctly caught) : {tp:,}")

test_metrics = {
    "f1_macro": round(f1, 4),
    "roc_auc":  round(auc, 4),
    "avg_precision": round(ap, 4),
    "recall_benign":    round(rec0, 4),
    "recall_sqli":      round(rec1, 4),
    "precision_benign": round(pre0, 4),
    "precision_sqli":   round(pre1, 4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — CONFUSION MATRIX PLOT")
print("=" * 60)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues",
    xticklabels=["Benign (pred)", "SQLi (pred)"],
    yticklabels=["Benign (true)", "SQLi (true)"],
    linewidths=0.5, ax=ax,
    annot_kws={"size": 14, "weight": "bold"},
)
ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Actual", fontsize=11)
ax.set_xlabel("Predicted", fontsize=11)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
plt.savefig(path)
plt.show()
print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ROC curve
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — ROC CURVE")
print("=" * 60)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4A90D9", lw=2, label=f"Linear SVM  (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="#4A90D9")
ax.set_xlabel("False Positive Rate  (1 − Specificity)")
ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)")
ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.02])
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "roc_curve.png")
plt.savefig(path)
plt.show()
print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Precision-Recall curve
#
# ⚠  CLASS IMBALANCE NOTE:
#    The PR curve is more informative than ROC when classes are imbalanced.
#    In production (99%+ benign), you'll want high precision to avoid flooding
#    security teams with false alerts. Track this curve as traffic shifts.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — PRECISION-RECALL CURVE")
print("=" * 60)

prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_curve, prec_curve, color="#E05C5C", lw=2,
        label=f"Linear SVM  (AP = {ap:.4f})")
ax.axhline(y_test.mean(), color="gray", linestyle="--", lw=1,
           label=f"Baseline (class prevalence = {y_test.mean():.2f})")
ax.fill_between(rec_curve, prec_curve, alpha=0.08, color="#E05C5C")
ax.set_xlabel("Recall  (SQLi sensitivity)")
ax.set_ylabel("Precision  (attack signal purity)")
ax.set_title("Precision-Recall Curve — Test Set", fontsize=13, fontweight="bold")
ax.legend(loc="lower left")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0.0, 1.02])
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "precision_recall_curve.png")
plt.savefig(path)
plt.show()
print(f"Saved: {path}")
print("  ⚠  In production (99%+ benign), monitor this curve — precision")
print("     can degrade sharply as benign queries dominate traffic.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SHAP feature importance
#
# Uses the inner LinearSVC's coefficients via LinearExplainer.
# Explains which features push the model toward SQLi vs Benign.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — SHAP FEATURE IMPORTANCE")
print("=" * 60)

print("  Computing SHAP values on 500-sample subset...")

# Use inner LinearSVC (pre-calibration) for LinearExplainer
inner_svc = model.calibrated_classifiers_[0].estimator
sample_idx = np.random.RandomState(42).choice(X_test.shape[0], 500, replace=False)
X_sample   = X_test[sample_idx]

explainer  = shap.LinearExplainer(inner_svc, X_sample)
shap_vals  = explainer.shap_values(X_sample)   # shape: (500, 10131)

# Mean absolute SHAP per feature
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top_idx       = np.argsort(mean_abs_shap)[::-1][:20]
top_names     = [feature_names[i] for i in top_idx]
top_scores    = mean_abs_shap[top_idx]

# Clean up TF-IDF prefix for display
top_names_clean = [n.replace("tfidf_", "tfidf: ").replace("kw_", "kw: ")
                    .replace("sc_count_", "sc: ").replace("sc_ratio_", "sc_ratio: ")
                    .replace("_", " ") for n in top_names]

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#4A90D9" if "tfidf" in n else "#E05C5C" if "kw" in n
          else "#F5A623" if "sc" in n else "#7B61FF"
          for n in top_names]
bars = ax.barh(range(20), top_scores[::-1], color=colors[::-1], alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(top_names_clean[::-1], fontsize=9)
ax.set_xlabel("Mean |SHAP value|  (impact on model output)")
ax.set_title("Top 20 Features by SHAP Importance", fontsize=13, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4A90D9", label="TF-IDF token"),
    Patch(facecolor="#E05C5C", label="SQL keyword count"),
    Patch(facecolor="#F5A623", label="Special char"),
    Patch(facecolor="#7B61FF", label="Structural / entropy"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "shap_top_features.png")
plt.savefig(path)
plt.show()
print(f"Saved: {path}")
print(f"\n  Top 5 most influential features:")
for i in range(5):
    print(f"    {i+1}. {top_names_clean[i]:<40}  mean|SHAP|={top_scores[i]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Error analysis (false positives & false negatives)
#
# Manually inspecting misclassified samples is the fastest way to spot
# gaps in your feature set or mislabelled training data.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — ERROR ANALYSIS")
print("=" * 60)

test_df = test_df.reset_index(drop=True)
test_df["y_pred"] = y_pred
test_df["y_prob"] = y_prob

# False Positives — benign queries flagged as SQLi
fp_df = test_df[(test_df["Label"] == 0) & (test_df["y_pred"] == 1)].copy()
fp_df = fp_df.sort_values("y_prob", ascending=False)

# False Negatives — SQLi attacks that slipped through
fn_df = test_df[(test_df["Label"] == 1) & (test_df["y_pred"] == 0)].copy()
fn_df = fn_df.sort_values("y_prob", ascending=True)

print(f"\nFalse Positives (benign → flagged as SQLi) : {len(fp_df):,}")
print(f"False Negatives (SQLi  → missed)           : {len(fn_df):,}")

print(f"\n--- Top 10 False Positives (highest SQLi confidence) ---")
for i, row in fp_df.head(10).iterrows():
    print(f"  prob={row['y_prob']:.3f} | {str(row['Query'])[:120]}")

print(f"\n--- Top 10 False Negatives (lowest SQLi confidence) ---")
for i, row in fn_df.head(10).iterrows():
    print(f"  prob={row['y_prob']:.3f} | {str(row['Query'])[:120]}")

# Confidence distribution plot for errors
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, df, title, color in [
    (axes[0], fp_df, f"False Positives (n={len(fp_df):,})\nBenign flagged as SQLi", "#F5A623"),
    (axes[1], fn_df, f"False Negatives (n={len(fn_df):,})\nSQLi that slipped through", "#E05C5C"),
]:
    if len(df):
        ax.hist(df["y_prob"], bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0.5, color="gray", linestyle="--", lw=1, label="threshold = 0.5")
        ax.set_xlabel("Model confidence (P(SQLi))")
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No errors!", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="green")
        ax.set_title(title, fontweight="bold")

plt.suptitle("Error Analysis — Confidence Distribution of Misclassified Samples",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "error_analysis.png")
plt.savefig(path)
plt.show()
print(f"\nSaved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save final test metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — SAVING TEST METRICS")
print("=" * 60)

metrics_path = os.path.join(REPORTS_DIR, "test_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"Saved: {metrics_path}")
print(json.dumps(test_metrics, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — EVALUATION SUMMARY")
print("=" * 60)
print(f"""
Model      : Linear SVM (CalibratedClassifierCV)
Test set   : {len(y_test):,} samples  ({(y_test==0).sum():,} benign / {(y_test==1).sum():,} SQLi)

Metric                  Value
──────────────────────────────
F1-macro                {f1:.4f}
ROC-AUC                 {auc:.4f}
Average Precision       {ap:.4f}
Recall  — Benign        {rec0:.4f}   {"✓" if rec0 >= 0.85 else "⚠  BELOW 0.85"}
Recall  — SQLi          {rec1:.4f}
Precision — Benign      {pre0:.4f}
Precision — SQLi        {pre1:.4f}

Confusion matrix:
  TN (correct benign)  : {tn:,}
  FP (false alarms)    : {fp:,}
  FN (missed attacks)  : {fn:,}
  TP (caught attacks)  : {tp:,}

Figures saved to reports/figures/:
  confusion_matrix.png
  roc_curve.png
  precision_recall_curve.png
  shap_top_features.png
  error_analysis.png

""")