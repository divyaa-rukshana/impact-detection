import argparse
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             precision_recall_curve, classification_report)
from preprocess import load_all, class_weights
from models import HIKNetLike, ConvBiGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FALL = 2


def predict(model, x, bs=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            #outs.append(model(torch.from_numpy(x[i:i + bs])))
            outs.append(model(torch.from_numpy(x[i:i + bs]).to(device)))
    return torch.cat(outs)


def binary_report(name, y_true, prob):
    pred = prob >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    p, r, _ = precision_recall_curve(y_true, prob)
    p_at_95 = p[r >= 0.95].max() if (r >= 0.95).any() else float("nan")
    print(f"{name}: sensitivity={tp/(tp+fn):.3f}  specificity={tn/(tn+fp):.3f}  "
          f"precision={tp/(tp+fp+1e-9):.3f}  AUC-ROC={roc_auc_score(y_true, prob):.3f}  "
          f"AUC-PR={average_precision_score(y_true, prob):.3f}  "
          f"precision@95%recall={p_at_95:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hiknet", "gru"], default="hiknet")
    ap.add_argument("--task", choices=["binary", "three"], default="binary")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    xtr, ytr, xva, yva, xte, yte = load_all(standardize=True)

    if a.task == "binary":
        n_out = 1
        ytr_t, yva_t = (ytr == FALL).astype(np.float32), (yva == FALL).astype(np.float32)
        pos_w = torch.tensor((ytr_t == 0).sum() / (ytr_t == 1).sum(), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        fix = lambda out: out.squeeze(1)
    else:
        n_out = 3
        ytr_t, yva_t = ytr.astype(np.int64), yva.astype(np.int64)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(ytr), dtype=torch.float32).to(device))
        fix = lambda out: out

    model = (HIKNetLike if a.model == "hiknet" else ConvBiGRU)(n_out=n_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr_t)),
                        batch_size=a.bs, shuffle=True)
    yva_tensor = torch.from_numpy(yva_t).to(device)

    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(fix(model(xb)), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        with torch.no_grad():
            val_loss = criterion(fix(predict(model, xva)), yva_tensor).item()
        print(f"epoch {ep:2d} | train {run/len(xtr):.4f} | val {val_loss:.4f} | {time.time()-t0:.0f}s")
        if val_loss < best_loss:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    os.makedirs("results", exist_ok=True)
    tag = f"{a.model}_{a.task}_seed{a.seed}"
    torch.save(best_state, f"results/{tag}.pt")

    for name, x, y in [("val ", xva, yva), ("test", xte, yte)]:
        out = predict(model, x)
        if a.task == "binary":
            prob = torch.sigmoid(out.squeeze(1)).cpu().numpy()
            binary_report(name, y == FALL, prob)
            if name.strip() == "test":
                np.save(f"results/{tag}_test_scores.npy", prob)
        else:
            prob = torch.softmax(out, dim=1).cpu().numpy()
            print(f"\n{name} (3-class), fall-vs-rest AUC-PR = "
                  f"{average_precision_score(y == FALL, prob[:, FALL]):.3f}")
            print(classification_report(y, prob.argmax(1),
                                        target_names=["nonfall", "prefall", "fall"], digits=3))


if __name__ == "__main__":
    main()