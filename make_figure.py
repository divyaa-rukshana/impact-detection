import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from preprocess import load_all
from baseline_svm import features

FALL = 2
xtr, ytr, xva, yva, xte, yte = load_all(standardize=False)
y = (yte == FALL)

# --- SVM baseline (refit, since its scores were not saved) ---
Ftr, Fte = features(xtr), features(xte)
sc = StandardScaler().fit(Ftr)
svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(Ftr), ytr == FALL)
svm_scores = svm.decision_function(sc.transform(Fte))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
style = {"hiknet": ("tab:blue", "HIKNet-style CNN (paper)"),
         "gru": ("tab:orange", "Conv + BiGRU (ours)")}

for model, (color, label) in style.items():
    files = sorted(glob.glob(f"results/{model}_binary_seed*_test_scores.npy"))
    for i, f in enumerate(files):
        s = np.load(f)
        p, r, _ = precision_recall_curve(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        lab = f"{label} ({len(files)} seeds)" if i == 0 else None
        ax1.plot(r, p, color=color, alpha=0.5, lw=1.2, label=lab)
        ax2.plot(fpr, tpr, color=color, alpha=0.5, lw=1.2, label=lab)

p, r, _ = precision_recall_curve(y, svm_scores)
print("SVM precision@95%recall:", round(p[r >= 0.95].max(), 3))
fpr, tpr, _ = roc_curve(y, svm_scores)
ax1.plot(r, p, color="gray", lw=1.8, label="Linear SVM + features")
ax2.plot(fpr, tpr, color="gray", lw=1.8, label="Linear SVM + features")

ax1.axvline(0.95, color="k", ls="--", lw=0.8)
ax1.text(0.951, 0.45, "95% recall", fontsize=8, rotation=90, va="bottom")
ax1.set(xlabel="Recall (sensitivity)", ylabel="Precision", ylim=(0.4, 1.02),
        title="Precision-recall curve (test set, fall = positive)")
ax2.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 0.1),
        ylim=(0.8, 1.005), title="ROC curve (zoomed to FPR ≤ 0.1)")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig("results/curves.png", dpi=200)
print("saved results/curves.png")