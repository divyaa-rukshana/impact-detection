import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score, classification_report)
from preprocess import load_all

FALL = 2


def features(x):
    """x: (N, 256, 6) -> (N, F) hand-crafted features."""
    acc = np.linalg.norm(x[:, :, :3], axis=2)   # accel magnitude over time
    gyr = np.linalg.norm(x[:, :, 3:], axis=2)   # gyro magnitude over time
    feats = []
    for sig in (acc, gyr):
        feats += [sig.max(1), sig.min(1), sig.mean(1), sig.std(1),
                  sig.argmax(1) / sig.shape[1]]           # where the peak occurs
        f = np.abs(np.fft.rfft(sig, axis=1))
        n = f.shape[1]
        feats += [f[:, :n // 8].sum(1), f[:, n // 8:n // 4].sum(1),
                  f[:, n // 4:].sum(1)]                   # low / mid / high band energy
    # per-channel max, min, std
    feats += [x.max(1), x.min(1), x.std(1)]
    out = [f if f.ndim == 2 else f[:, None] for f in feats]
    return np.concatenate(out, axis=1)


def report(name, y_true, score, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"{name}: sensitivity={tp/(tp+fn):.3f}  specificity={tn/(tn+fp):.3f}  "
          f"precision={tp/(tp+fp+1e-9):.3f}  AUC-ROC={roc_auc_score(y_true, score):.3f}  "
          f"AUC-PR={average_precision_score(y_true, score):.3f}")


if __name__ == "__main__":
    xtr, ytr, xva, yva, xte, yte = load_all(standardize=False)
    btr, bva, bte = (ytr == FALL), (yva == FALL), (yte == FALL)   # binary labels

    Ftr, Fva, Fte = features(xtr), features(xva), features(xte)
    sc = StandardScaler().fit(Ftr)
    Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
    print("feature matrix:", Ftr.shape)

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    clf.fit(Ftr, btr)

    for name, F, b in [("val ", Fva, bva), ("test", Fte, bte)]:
        report(name, b, clf.decision_function(F), clf.predict(F))
    print()
    print(classification_report(bte, clf.predict(Fte), target_names=["not fall", "fall"]))