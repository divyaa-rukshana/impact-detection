import os
import numpy as np

D = os.path.join("data", "Three Classes")
T, C = 256, 6


def load_split(split):
    y = np.fromfile(os.path.join(D, f"y_{split}_3"), dtype=np.uint8).reshape(-1, 3).argmax(axis=1)
    x = np.fromfile(os.path.join(D, f"x_{split}_3"), dtype=np.float32).reshape(len(y), T, C)
    return x, y


def load_all(standardize=True):
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")
    if standardize:
        # statistics from the training set only, to avoid leakage
        mean = xtr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = (xtr.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        for a in (xtr, xva, xte):
            a -= mean   # in place, saves memory
            a /= std
    return xtr, ytr, xva, yva, xte, yte


def class_weights(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes)
    return counts.max() / counts  # majority class gets 1.0


if __name__ == "__main__":
    # figure out which class is which
    x, y = load_split("train")
    acc = np.linalg.norm(x[:, :, :3], axis=2)
    gyr = np.linalg.norm(x[:, :, 3:], axis=2)
    for c in range(3):
        m = y == c
        print(f"class {c}: n={m.sum():6d} | mean peak accel={acc[m].max(axis=1).mean():.3f}"
              f" | mean peak gyro={gyr[m].max(axis=1).mean():.3f}")
    print("computed class weights:", class_weights(y).round(2))