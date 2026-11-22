"""
ATM/UPI Anomaly Detection - Steps 1 to 6 as one runnable script (scales to 1M+ transactions).

HOW TO RUN IN VS CODE
  * Plain run:  python main.py          (prints results, saves charts into ./plots)
  * Cell by cell: install the 'Python' + 'Jupyter' extensions, then click 'Run Cell'
    above any '# %%' line. Charts then appear in the Interactive Window.
Needs roughly 3 GB of free RAM at the default size. Lower N_USERS / DAYS if needed.
"""
# %% Imports & setup
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import joblib
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from anomaly_core import *

N_USERS, DAYS = 12_000, 30 #1 million transactions.
MAX_TRAIN = 200_000

os.makedirs("plots", exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
T0 = time.time()

def show(name):
    """Save chart to ./plots; display it only when running in an interactive window."""
    plt.tight_layout(); plt.savefig(f"plots/{name}.png", dpi=120)
    plt.show() if "inline" in matplotlib.get_backend().lower() else plt.close()

def title(t): print("\n" + "=" * 70 + f"\n{t}   [{time.time() - T0:.0f}s elapsed]\n" + "=" * 70)

# %% STEP 1
title("STEP 1: Why unsupervised?")
print("""Fraud labels are scarce because:
 - fraud is <0.1% of transactions (extreme class imbalance)
 - labels arrive late (complaints/chargebacks) and many frauds are never reported
 - fraud tactics change (concept drift), so old labelled fraud misses new patterns
 - labelling needs human investigators and the data is privacy-sensitive
=> model what NORMAL looks like, flag what deviates.""")

# %% Create data(synthetic)
title("DATA: synthetic transactions")
users = make_users(N_USERS, np.random.default_rng(0))
raw = generate(users, "2026-01-01", days=DAYS, anomaly_rate=0.01, seed=0)
print(f"{len(raw):,} transactions | injected anomaly share: {raw.label.mean():.4f}")
print(raw.head())

# %% STEP 2 Feature engineering
title("STEP 2: Feature engineering")
F = build_features(raw, users)
X = F[FEATURES].values
print(F[FEATURES].describe().T[["mean", "std", "min", "max"]].round(2))

fig, ax = plt.subplots(1, 4, figsize=(16, 3))
for a, col in zip(ax, ["amount_vs_user", "log_gap_s", "speed_kmh_log", "txn_count_1h"]):
    a.hist(F.loc[F.label == 0, col], bins=40, alpha=.7, density=True, label="normal")
    a.hist(F.loc[F.label == 1, col], bins=40, alpha=.7, density=True, label="injected (hidden)")
    a.set_title(col)
ax[0].legend(); show("01_feature_distributions")

# %% STEP 3 Train Isolation Forest + autoencoder (on a sample), score everything
title(f"STEP 3: Train models on a {MAX_TRAIN:,}-row sample (no labels used), score all {len(F):,}")
det = Detector(alert_rate=0.01, max_train=MAX_TRAIN).fit(X)
BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
joblib.dump(det, os.path.join(BASE, "detector.joblib"))     # app.py loads this same model
print("Saved trained detector -> detector.joblib")
S = det.score(X)
F["score"], F["iso"], F["ae"] = S["score"], S["iso"], S["ae"]
print("Trained Isolation Forest + autoencoder. Combined score = mean percentile rank of both.")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
ax[0].hist(F.iso, bins=60); ax[0].set_title("Isolation Forest anomaly score")
ax[1].hist(np.log10(F.ae), bins=60); ax[1].set_title("Autoencoder log10(reconstruction error)")
show("02_score_distributions")

# %% STEP 4 Threshold
title("STEP 4: Threshold selection (alert budget)")
thr = det.threshold
F["flag"] = F.score >= thr
print(f"threshold = {thr:.3f} -> flagged {F.flag.sum():,} of {len(F):,} ({F.flag.mean():.2%})")
for q in [.95, .97, .98, .99, .995]:
    c = np.quantile(F.score, q); print(f"  top {1-q:>5.1%} -> score >= {c:.3f} ({int((F.score >= c).sum()):,} alerts)")

s = np.sort(F.score.values)[::-1]
plt.figure(figsize=(7, 3)); plt.plot(s[:int(.05 * len(s))]); plt.axhline(thr, color="r", ls="--", label="chosen threshold")
plt.xlabel("rank"); plt.ylabel("score"); plt.legend(); show("03_threshold_elbow")

# %% STEP 5 Inspect top anomalies
title("STEP 5: Top anomalies and why they look suspicious")
top = F[F.flag].sort_values("score", ascending=False).head(15).copy()
top["why_suspicious"] = [", ".join(explain(r)) for r in top.itertuples()]
print(top[["ts", "user_id", "amount", "channel", "city", "home_city", "score", "why_suspicious"]].round(2).to_string(index=False))

err = pd.DataFrame(S["ae_feat"], columns=FEATURES)[F.flag.values]
plt.figure(figsize=(7, 3)); err.idxmax(axis=1).value_counts().plot.barh(title="Feature driving autoencoder error (flagged txns)")
show("04_ae_feature_blame")

# %% STEP 6 Validation without labels
title("STEP 6: Validation without ground truth")
k = int(.01 * len(F))
print("(a) Model agreement: Spearman IF vs AE =", round(spearmanr(F.iso, F.ae)[0], 3))
a, b = set(F.nlargest(k, "iso").index), set(F.nlargest(k, "ae").index)
print(f"    Top-1% overlap IF vs AE: {len(a & b) / k:.0%}")

ev = np.random.default_rng(0).choice(len(X), min(100_000, len(X)), replace=False); k_ev = int(.01 * len(ev))
tops = [set(np.argsort(-Detector(seed=sd, max_train=MAX_TRAIN).fit(X).score(X[ev])["score"])[:k_ev]) for sd in (1, 2, 3)]
print("(b) Stability: top-1% overlap between seeds:", [f"{len(tops[0] & t) / k_ev:.0%}" for t in tops[1:]])

hold = users.sample(min(3000, len(users)), random_state=1)
raw2 = generate(hold, "2026-03-01", DAYS, 0.01, seed=99); F2 = build_features(raw2, hold)
S2 = det.score(F2[FEATURES].values); fl = S2["score"] >= thr
print(f"(c) Held-out injection test ({len(F2):,} txns): {fl.sum():,} alerts | precision {F2.label[fl].mean():.2f} | recall {F2.label[fl].sum() / F2.label.sum():.2f}")
print("    Recall by fraud type:")
print(F2[F2.label == 1].assign(hit=fl[F2.label.values == 1]).groupby("kind").hit.mean().round(2).to_string())

print("(d) Ground-truth metrics (possible ONLY because data is synthetic):")
print(pd.DataFrame([(n, roc_auc_score(F.label, v), average_precision_score(F.label, v))
      for n, v in [("Isolation Forest", S["iso"]), ("Autoencoder", S["ae"]), ("Combined", S["score"])]],
      columns=["model", "ROC-AUC", "PR-AUC"]).round(3).to_string(index=False))

p, r, _ = precision_recall_curve(F.label, F.score)
plt.figure(figsize=(6, 3)); plt.plot(r, p); plt.xlabel("recall"); plt.ylabel("precision"); plt.title("PR curve (synthetic truth)")
show("05_pr_curve")

# %% STEP 6e Accuracy metrics (needs the synthetic ground truth)
title("STEP 6e: Accuracy metrics on the held-out month")
y, yhat = F2.label.values, fl.astype(int)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
print(f"Confusion matrix: TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
print(f"Accuracy : {accuracy_score(y, yhat):.3f}   <- misleading: predicting 'all normal' scores {1 - y.mean():.3f}")
print(f"Precision: {precision_score(y, yhat):.3f}   (of alerts, share that are real fraud)")
print(f"Recall   : {recall_score(y, yhat):.3f}   (of real fraud, share caught)")
print(f"F1       : {f1_score(y, yhat):.3f}")
print(f"Balanced accuracy: {(recall_score(y, yhat) + tn / (tn + fp)) / 2:.3f}")
print(f"ROC-AUC  : {roc_auc_score(y, S2['score']):.3f} | PR-AUC: {average_precision_score(y, S2['score']):.3f}")
print(f"Fraud share of rows: {y.mean():.2%} | alert budget: {fl.mean():.2%}  -> max possible recall ~ {min(1, fl.mean() / y.mean()):.2f}")

print("\nPrecision@k / Recall@k (rank by score, review top-k):")
order = np.argsort(-S2["score"])
for kk in [100, 500, 1000, 2500]:
    hit = y[order[:kk]].sum(); print(f"  k={kk:>5,}: precision {hit / kk:.2f} | recall {hit / y.sum():.2f}")

print("\nEffect of alert budget (re-thresholding the same scores):")
for q in [.005, .01, .02, .03, .05]:
    t = np.quantile(S["score"], 1 - q); f_ = S2["score"] >= t
    print(f"  top {q:>5.1%}: precision {precision_score(y, f_):.2f} | recall {recall_score(y, f_):.2f} | F1 {f1_score(y, f_):.2f}")

# %% Save flagged transactions
F[F.flag].sort_values("score", ascending=False).to_csv("flagged_transactions.csv", index=False)
print(f"\nSaved flagged_transactions.csv ({int(F.flag.sum()):,} rows) and charts in ./plots  [{time.time() - T0:.0f}s total]")