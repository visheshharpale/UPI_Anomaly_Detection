"""Run:  python app.py   ->  open http://127.0.0.1:5000"""
import os, warnings, joblib; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, render_template
from anomaly_core import *

app = Flask(__name__)
USERS = make_users(150, np.random.default_rng(0))
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector.joblib")
if os.path.exists(MODEL_PATH):
    DET = joblib.load(MODEL_PATH)          # the SAME model trained & evaluated by main.py
    print("Loaded detector.joblib (trained by main.py)")
else:                                      # fallback so the demo still runs on its own
    print("detector.joblib not found - training a small detector (run main.py first to use the full model) ...")
    train_df = build_features(generate(USERS, "2026-01-01", 14, 0.01, seed=0), USERS)
    DET = Detector(alert_rate=0.02).fit(train_df[FEATURES].values)
print("Ready. Alert threshold (percentile score) =", round(float(DET.threshold), 3))
LIVE_START = pd.Timestamp("2026-02-01")

def top_feature(fe):  # which feature the autoencoder reconstructs worst
    return FEATURES[int(np.argmax(fe))]

def package(F, s, keep):
    out = []
    for i in np.where(keep)[0]:
        r = F.iloc[i]
        out.append(dict(ts=r.ts.strftime("%Y-%m-%d %H:%M:%S"), user=int(r.user_id), home=r.home_city,
            city=r.city, channel=r.channel, amount=round(float(r.amount), 2), score=round(float(s["score"][i]), 4),
            iso=round(float(s["iso"][i]), 3), ae=round(float(s["ae"][i]), 3), ae_top=top_feature(s["ae_feat"][i]),
            reasons=explain(r), truth=r.kind))
    return out

@app.route("/")
def index(): return render_template("index.html", cities=CITY_NAMES, threshold=float(DET.threshold))

@app.post("/api/simulate")
def simulate():
    p = request.get_json(force=True)
    rate, seed, n_users = float(p.get("rate", .03)), int(p.get("seed", 1)), int(p.get("users", 40))
    u = USERS.sample(n_users, random_state=seed).reset_index(drop=True) if n_users < len(USERS) else USERS
    # 1 day of 'context' history + 1 live day; fraud only injected in the live day
    df = generate(u, LIVE_START - pd.Timedelta(days=1), 2, rate, anom_after=LIVE_START, seed=seed)
    F = build_features(df, u); s = DET.score(F[FEATURES].values)
    keep = (F.ts >= LIVE_START).values
    return jsonify(txns=package(F, s, keep), threshold=float(DET.threshold))

@app.post("/api/manual")
def manual():
    p = request.get_json(force=True); uid = int(p["user"])
    u = USERS[USERS.user_id == uid]
    hist = generate(u, LIVE_START - pd.Timedelta(days=3), 3, 0, seed=uid)
    h, m = map(int, p["time"].split(":")); t0 = LIVE_START + pd.Timedelta(hours=h, minutes=m)
    rows = [[uid, t0 + pd.Timedelta(seconds=40*j), float(p["amount"]), p["channel"], p["city"], 0, "manual"]
            for j in range(int(p.get("repeat", 1)))]
    df = pd.concat([hist, pd.DataFrame(rows, columns=hist.columns)], ignore_index=True)
    F = build_features(df, u); s = DET.score(F[FEATURES].values)
    return jsonify(txns=package(F, s, (F.ts >= LIVE_START).values), threshold=float(DET.threshold),
                   profile=dict(home=u.home.iloc[0], usual_amount=round(float(u.mean_amt.iloc[0]))))

@app.get("/api/user/<int:uid>")
def user(uid):
    u = USERS[USERS.user_id == uid].iloc[0]
    return jsonify(home=u.home, usual_amount=round(float(u.mean_amt)), peak_hour=round(float(u.peak_hour), 1))

if __name__ == "__main__": app.run(debug=False)