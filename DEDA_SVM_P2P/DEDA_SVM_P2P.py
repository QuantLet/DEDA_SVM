# ============================================================
# Python translation of original R script (faithful version)
# Probabilities: RBF SVM
# Class prediction: Polynomial SVM
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
import statsmodels.api as sm

# ------------------------------------------------------------
# Read data
# ------------------------------------------------------------
data = pd.read_csv("/mnt/data/p2p.csv")

# Remove first column (ID column like in R)
data = data.iloc[:, 1:]

# Convert status to factor
data["status"] = data["status"].astype("category")

# ------------------------------------------------------------
# Train / test split (75%)
# ------------------------------------------------------------
X = data.drop(columns="status")
y = data["status"].cat.codes   # factor -> 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1234,
    stratify=y
)

# ------------------------------------------------------------
# Variables used (same as R file)
# ------------------------------------------------------------
vars_used = [
    "ratio003", "ratio004", "ratio005", "ratio006",
    "ratio011", "ratio012", "DPO", "DSO", "turnover",
    "ratio036", "ratio037", "ratio039", "ratio040"
]

# ------------------------------------------------------------
# Logistic regression
# ------------------------------------------------------------
X_train_log = sm.add_constant(X_train[vars_used])
X_test_log = sm.add_constant(X_test[vars_used])

fit_log = sm.Logit(y_train, X_train_log).fit()
pre_log = fit_log.predict(X_test_log)

class_log = (pre_log > 0.5).astype(int)

# ------------------------------------------------------------
# SVM for probabilistic prediction (RBF kernel)
# ------------------------------------------------------------
svm_rbf = SVC(
    kernel="rbf",
    probability=True,   # needed for ROC/AUC/Brier
    C=1,
    gamma="scale"
)

svm_rbf.fit(X_train[vars_used], y_train)

# Probabilities (class 1)
pre_svm = svm_rbf.predict_proba(X_test[vars_used])[:, 1]

# ------------------------------------------------------------
# SVM for class prediction (polynomial kernel)
# ------------------------------------------------------------
svm_poly = SVC(
    kernel="poly",
    degree=3,
    C=1
)

svm_poly.fit(X_train[vars_used], y_train)

class_svm = svm_poly.predict(X_test[vars_used])

# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------
print("Polynomial SVM Confusion Matrix")
print(confusion_matrix(y_test, class_svm))

print("\nLogistic Regression Confusion Matrix")
print(confusion_matrix(y_test, class_log))

# ------------------------------------------------------------
# ROC curves
# ------------------------------------------------------------
fpr_log, tpr_log, _ = roc_curve(y_test, pre_log)
fpr_svm, tpr_svm, _ = roc_curve(y_test, pre_svm)

# Logistic ROC
plt.figure(figsize=(6, 6))
plt.plot(fpr_log, tpr_log)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Logistic ROC (AUC = {roc_auc_score(y_test, pre_log):.3f})")
plt.savefig("logit.jpg")
plt.close()

# SVM ROC (RBF probabilities)
plt.figure(figsize=(6, 6))
plt.plot(fpr_svm, tpr_svm)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"SVM ROC (AUC = {roc_auc_score(y_test, pre_svm):.3f})")
plt.savefig("svm.jpg")
plt.close()

# ------------------------------------------------------------
# AUC
# ------------------------------------------------------------
auc_log = roc_auc_score(y_test, pre_log)
auc_svm = roc_auc_score(y_test, pre_svm)

print("\nAUC")
print("Logistic:", auc_log)
print("SVM (RBF prob):", auc_svm)

# ------------------------------------------------------------
# Brier scores
# ------------------------------------------------------------
brier_log = np.mean((pre_log - y_test) ** 2)
brier_svm = np.mean((pre_svm - y_test) ** 2)

print("\nBrier Scores")
print("Logistic:", brier_log)
print("SVM:", brier_svm)
