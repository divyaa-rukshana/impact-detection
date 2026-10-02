import glob
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from preprocess import load_split

_, yte = load_split("test")
y = (yte == 2)

rows = defaultdict(list)
for f in sorted(glob.glob("results/*_binary_seed*_test_scores.npy")):
    model = re.search(r"results[/\\](\w+?)_binary", f).group(1)
    s = np.load(f)
    p, r, _ = precision_recall_curve(y, s)
    p95 = p[r >= 0.95].max()
    rows[model].append((average_precision_score(y, s), roc_auc_score(y, s), p95))

lines = []
for model, vals in rows.items():
    v = np.array(vals)
    m, sd = v.mean(0), v.std(0)
    line = (f"{model:7s} runs={len(v)} | AUC-PR {m[0]:.3f}±{sd[0]:.3f} | "
            f"AUC-ROC {m[1]:.3f}±{sd[1]:.3f} | prec@95%rec {m[2]:.3f}±{sd[2]:.3f}")
    print(line)
    lines.append(line)
open("results/summary.txt", "w").write("\n".join(lines))