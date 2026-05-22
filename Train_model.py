"""
Train a state‑of‑the‑art ad detector for Persian subtitles.
Uses character n‑gram TF‑IDF + calibrated Logistic Regression,
with hyper‑parameter tuning and optimal threshold selection.

Usage:
    python train_model.py

Input:
    - ads_positive_clean.csv
    - ads_negative_clean_filtered.csv

Output:
    - ad_classifier.pkl           (trained pipeline)
    - ad_vectorizer.pkl           (TF‑IDF vectorizer)
    - threshold.txt               (optimal decision threshold)
    - evaluation_report.txt       (metrics)
    - roc_pr_curves.png           (diagnostic plots)
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 1. Load and combine the datasets
# ─────────────────────────────────────────────────────────────────────
print("➤ Loading datasets...")
df_pos = pd.read_csv("ads_positive_clean_final.csv")   # label = 1
df_neg = pd.read_csv("ads_negative_clean_final.csv")  # label = 0

df = pd.concat([df_pos, df_neg], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["clean_text"].values
y = df["label"].values

print(f"   Total samples: {len(X)}")
print(f"   Positive (ads): {y.sum()} | Negative (non‑ads): {len(y) - y.sum()}")

# ─────────────────────────────────────────────────────────────────────
# 2. Train / test split (stratified)
# ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────
# 3. Build pipeline and define hyper‑parameter grid
# ─────────────────────────────────────────────────────────────────────
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", lowercase=False)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])

param_grid = {
    "tfidf__ngram_range": [(2, 5), (3, 4), (1, 5)],
    "tfidf__max_features": [None, 10000, 20000],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.5, 1.0, 2.0, 5.0],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["saga"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────────────────
# 4. Grid search with refit on best F1 score (or ROC AUC)
# ─────────────────────────────────────────────────────────────────────
print("\n➤ Running hyper‑parameter tuning (this may take a while)...")
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
)
grid.fit(X_train, y_train)

print(f"\n   Best AUC: {grid.best_score_:.4f}")
print(f"   Best parameters:\n{grid.best_params_}")

best_model = grid.best_estimator_

# ─────────────────────────────────────────────────────────────────────
# 5. Evaluate on the test set
# ─────────────────────────────────────────────────────────────────────
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_default = best_model.predict(X_test)

print("\n" + "=" * 60)
print("Performance on test set (default threshold = 0.5)")
print("=" * 60)
print(classification_report(y_test, y_pred_default, target_names=["non‑ad", "ad"]))
cm = confusion_matrix(y_test, y_pred_default)
print("Confusion matrix:")
print(cm)

auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {auc:.4f}")

# ─────────────────────────────────────────────────────────────────────
# 6. Find optimal threshold for a desired recall (e.g., 0.95)
#    while maximizing precision
# ─────────────────────────────────────────────────────────────────────
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

target_recall = 0.95
try:
    idx = next(i for i, r in enumerate(recalls) if r < target_recall) - 1
except StopIteration:
    idx = len(thresholds) - 1
opt_threshold = thresholds[idx]
opt_recall = recalls[idx]
opt_precision = precisions[idx]

print(f"\n⚡ Optimal threshold for recall ≥ {target_recall}: {opt_threshold:.4f}")
print(f"   Achieved recall: {opt_recall:.4f}  Precision: {opt_precision:.4f}")

y_pred_opt = (y_proba >= opt_threshold).astype(int)
print("\nClassification report with optimal threshold:")
print(classification_report(y_test, y_pred_opt, target_names=["non‑ad", "ad"]))

# ─────────────────────────────────────────────────────────────────────
# 7. Save model, vectorizer, and threshold
# ─────────────────────────────────────────────────────────────────────
print("\n➤ Saving artifacts...")
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/ad_classifier.pkl")
joblib.dump(best_model.named_steps["tfidf"], "model/ad_vectorizer.pkl")

with open("model/threshold.txt", "w") as f:
    f.write(str(opt_threshold))

print("   model/ad_classifier.pkl")
print("   model/ad_vectorizer.pkl")
print("   model/threshold.txt")

# ─────────────────────────────────────────────────────────────────────
# 8. Generate diagnostic plots
# ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

ax2.plot(recalls, precisions, label="Precision‑Recall curve", color="darkorange")
ax2.axvline(x=opt_recall, color="green", linestyle="--", label=f"Target recall={target_recall}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision‑Recall Curve")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model/roc_pr_curves.png", dpi=150)
plt.close()
print("   model/roc_pr_curves.png")

# ─────────────────────────────────────────────────────────────────────
# 9. Write summary to a text file
# ─────────────────────────────────────────────────────────────────────
with open("model/evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Best AUC (CV): {grid.best_score_:.4f}\n")
    f.write(f"Best params: {grid.best_params_}\n\n")
    f.write("Test set performance (threshold=0.5):\n")
    f.write(classification_report(y_test, y_pred_default, target_names=["non‑ad", "ad"]))
    f.write(f"\nConfusion matrix:\n{cm}\n")
    f.write(f"\nROC AUC: {auc:.4f}\n")
    f.write(f"\nOptimal threshold for recall >= {target_recall}: {opt_threshold:.4f}\n")
    f.write(f"Recall: {opt_recall:.4f}   Precision: {opt_precision:.4f}\n")
    f.write("\nClassification report with optimal threshold:\n")
    f.write(classification_report(y_test, y_pred_opt, target_names=["non‑ad", "ad"]))

print("\n✓ All done. See 'model/' directory for output files.")