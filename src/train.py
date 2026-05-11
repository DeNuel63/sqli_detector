"""
SQL Injection Detection — Model Selection & Training
=====================================================
Input  : data/processed/features_train.npz / labels_train.npy
         data/processed/features_val.npz   / labels_val.npy
Output : models/model.pkl                  ← best model
         reports/metrics.json              ← val scores for all candidates

Run:
    python src/train.py

⚠  CLASS IMBALANCE REMINDER (1.3:1 SQLi:Benign)
    All models use class_weight='balanced' or scale_pos_weight.
    After training, CHECK recall for class 0 (benign) on the val set.
    Target: recall(benign) >= 0.85
    If it falls short → enable SMOTE in Step 4 (instructions at bottom).
"""

import os
import json
import pickle
import warnings
import numpy as np
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.svm          import LinearSVC
from sklearn.calibration  import CalibratedClassifierCV
from sklearn.ensemble     import RandomForestClassifier
from sklearn.metrics      import (classification_report, confusion_matrix,
                                  roc_auc_score, f1_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ── Paths (aligned with agreed project structure) ─────────────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load feature matrices
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING FEATURE MATRICES")
print("=" * 60)

X_train = load_npz(os.path.join(PROCESSED_DIR, "features_train.npz"))
X_val   = load_npz(os.path.join(PROCESSED_DIR, "features_val.npz"))
X_test  = load_npz(os.path.join(PROCESSED_DIR, "features_test.npz"))
y_train = np.load(os.path.join(PROCESSED_DIR, "labels_train.npy"))
y_val   = np.load(os.path.join(PROCESSED_DIR, "labels_val.npy"))
y_test  = np.load(os.path.join(PROCESSED_DIR, "labels_test.npy"))

print(f"X_train : {X_train.shape}  |  Benign: {(y_train==0).sum():,}  SQLi: {(y_train==1).sum():,}")
print(f"X_val   : {X_val.shape}    |  Benign: {(y_val==0).sum():,}   SQLi: {(y_val==1).sum():,}")
print(f"X_test  : {X_test.shape}   |  Benign: {(y_test==0).sum():,}   SQLi: {(y_test==1).sum():,}")
print(f"Feature matrix sparsity : "
      f"{100*(1 - X_train.nnz/(X_train.shape[0]*X_train.shape[1])):.2f}%\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Define candidate models
#
# ⚠  CLASS IMBALANCE: all models use class_weight='balanced'.
#    This tells the model to treat each benign sample as ~1.3x more
#    important during training, compensating for the 57/43 ratio without
#    touching the data. If val recall(benign) < 0.85 after training,
#    switch to SMOTE resampling instead (see bottom of this file).
#
# Note on model choices:
#   - Logistic Regression : fast, strong baseline on sparse TF-IDF data
#   - Linear SVM          : often best for high-dimensional sparse text
#   - Random Forest       : captures non-linear keyword interactions
#   XGBoost excluded here — too slow for 10k features on CPU without GPU.
#   Add it back if you have a GPU or run overnight.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — DEFINING CANDIDATE MODELS")
print("=" * 60)

CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",         # fast + stable for dense feature sets
        random_state=42,
    ),
    "Logistic Regression (high C)": LogisticRegression(
        C=10.0,                 # less regularisation — better if data is clean
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
    "Linear SVM": CalibratedClassifierCV(   # wrapped for predict_proba + ROC-AUC
        LinearSVC(
            C=0.1,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    ),
}

for name in CANDIDATES:
    print(f"  ✓  {name}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — 5-fold stratified cross-validation (training set only)
#
# CV gives a stable generalisation estimate before the val set is touched.
# Stratified folds preserve the 57/43 class ratio in every fold.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — 5-FOLD STRATIFIED CROSS-VALIDATION (TRAIN SET)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in CANDIDATES.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    cv_results[name] = scores
    print(f"  {name:<35}  CV F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train all candidates on full train set, evaluate on val set
#
# Val set is used here for model selection ONLY — never for final scoring.
# Final test-set evaluation happens in evaluate.py.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 — TRAIN ON FULL TRAIN SET → EVALUATE ON VAL SET")
print("=" * 60)

val_results = {}

for name, model in CANDIDATES.items():
    print(f"\n--- {name} ---")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    f1   = f1_score(y_val, y_pred, average="macro")
    auc  = roc_auc_score(y_val, y_prob)
    rec0 = recall_score(y_val, y_pred, pos_label=0)   # benign recall ← watch this
    rec1 = recall_score(y_val, y_pred, pos_label=1)   # sqli recall

    val_results[name] = {
        "cv_f1_mean":        float(cv_results[name].mean()),
        "cv_f1_std":         float(cv_results[name].std()),
        "val_f1_macro":      float(f1),
        "val_roc_auc":       float(auc),
        "val_recall_benign": float(rec0),
        "val_recall_sqli":   float(rec1),
    }

    print(classification_report(y_val, y_pred, target_names=["Benign", "SQLi"]))
    print(f"  ROC-AUC          : {auc:.4f}")

    # ⚠  Class imbalance check — benign recall is the canary
    if rec0 < 0.85:
        print(f"  Recall (benign)  : {rec0:.4f}  ⚠  BELOW 0.85 — enable SMOTE (see bottom)")
    else:
        print(f"  Recall (benign)  : {rec0:.4f}  ✓")

    cm = confusion_matrix(y_val, y_pred)
    print(f"  Confusion matrix :")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}   (benign correctly/incorrectly classified)")
    print(f"    FN={cm[1,0]:,}   TP={cm[1,1]:,}  (SQLi missed / caught)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Select best model
#
# Primary   : highest val F1-macro  (balanced across both classes)
# Secondary : val recall(benign) >= 0.85  (guards against false alarm fatigue)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — MODEL SELECTION SUMMARY")
print("=" * 60)

print(f"\n{'Model':<35} {'CV F1':>7} {'Val F1':>7} {'AUC':>7} {'Rec(0)':>8} {'Rec(1)':>8}")
print("-" * 78)
for name, r in val_results.items():
    flag = "  ⚠" if r["val_recall_benign"] < 0.85 else ""
    print(f"{name:<35} {r['cv_f1_mean']:>7.4f} {r['val_f1_macro']:>7.4f} "
          f"{r['val_roc_auc']:>7.4f} {r['val_recall_benign']:>8.4f} "
          f"{r['val_recall_sqli']:>8.4f}{flag}")

eligible = {n: r for n, r in val_results.items() if r["val_recall_benign"] >= 0.85}
if not eligible:
    print("\n⚠  No model meets recall(benign) >= 0.85 — relaxing to pick by F1 only.")
    eligible = val_results

best_name  = max(eligible, key=lambda n: eligible[n]["val_f1_macro"])
best_model = CANDIDATES[best_name]

print(f"\n✓  Best model selected : {best_name}")
print(f"   Val F1-macro        : {val_results[best_name]['val_f1_macro']:.4f}")
print(f"   Recall (benign)     : {val_results[best_name]['val_recall_benign']:.4f}")
print(f"   ROC-AUC             : {val_results[best_name]['val_roc_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Save best model & all metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — SAVING MODEL & METRICS")
print("=" * 60)

# Best model → models/  (tfidf_vectorizer.pkl already saved there by features.py)
model_path = os.path.join(MODELS_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Saved: {model_path}  ← {best_name}")

# All candidate metrics → reports/
metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump({"best_model": best_name, "candidates": val_results}, f, indent=2)
print(f"Saved: {metrics_path}")

print(f"""
============================================================
TRAINING COMPLETE
============================================================
Best model : {best_name}
Saved to   : {model_path}

""")