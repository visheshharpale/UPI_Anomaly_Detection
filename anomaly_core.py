"""Core logic: synthetic data, feature engineering, models, scoring.
Shared by the notebook and the Flask app so both use identical code."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CITIES = {  # name: (lat, lon)
    "Mumbai": (19.07, 72.88), "Pune": (18.52, 73.86), "Delhi": (28.61, 77.21),
    "Bengaluru": (12.97, 77.59), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36),
    "Hyderabad": (17.39, 78.49), "Ahmedabad": (23.02, 72.57)}
CITY_NAMES = list(CITIES)
FEATURES = ["log_amount", "amount_vs_user", "log_gap_s", "hour_sin", "hour_cos",
            "dist_km", "speed_kmh_log", "dist_home_km", "txn_count_1h", "is_upi"]

def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 12742 * np.arcsin(np.sqrt(a))

def make_users(n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame({
        "user_id": np.arange(n),
        "home": rng.choice(CITY_NAMES, n, p=[.3,.15,.15,.12,.08,.07,.07,.06]),
        "mean_amt": rng.lognormal(6.3, .6, n),       # typical ticket size (Rs)
        "peak_hour": rng.normal(15, 3, n).clip(9, 21),
        "txn_per_day": rng.uniform(1, 5, n),
        "upi_share": rng.uniform(.3, .95, n)})

def generate(users, start, days=28, anomaly_rate=0.01, anom_after=None, seed=0):
    """Simulate normal behaviour + inject 4 fraud patterns. 'label' is kept ONLY for demo/sanity checks."""
    rng = np.random.default_rng(seed); rows = []
    start = pd.Timestamp(start); anom_after = pd.Timestamp(anom_after) if anom_after is not None else start
    for u in users.itertuples():
        for d in range(days):
            for _ in range(rng.poisson(u.txn_per_day)):
                h = float(np.clip(rng.normal(u.peak_hour, 3), 6, 23.5))
                ts = start + pd.Timedelta(days=d, hours=h)
                ch = "UPI" if rng.random() < u.upi_share else "ATM"
                rows.append([u.user_id, ts, rng.lognormal(np.log(u.mean_amt), .5), ch, u.home, 0, "normal"])
    df = pd.DataFrame(rows, columns=["user_id","ts","amount","channel","city","label","kind"])
    # inject anomalies
    pool = df[df.ts >= anom_after]; k = max(1, int(anomaly_rate*len(pool))) if anomaly_rate > 0 else 0
    new = []
    for i in rng.choice(pool.index, k, replace=False) if k else []:
        r = df.loc[i]; kind = rng.choice(["big_amount","odd_hour","location_jump","burst"])
        u = users.loc[users.user_id == r.user_id].iloc[0]
        if kind == "big_amount":
            df.loc[i, ["amount","kind","label"]] = [u.mean_amt*rng.uniform(12, 40), kind, 1]
        elif kind == "odd_hour":
            df.loc[i, "ts"] = r.ts.normalize() + pd.Timedelta(hours=rng.uniform(2, 4))
            df.loc[i, ["amount","kind","label"]] = [u.mean_amt*rng.uniform(4, 9), kind, 1]
        elif kind == "location_jump":
            far = rng.choice([c for c in CITY_NAMES if c != u.home])
            df.loc[i, ["city","kind","label"]] = [far, kind, 1]
            df.loc[i, "ts"] = r.ts + pd.Timedelta(minutes=rng.uniform(5, 40))
        else:  # burst of rapid small withdrawals/payments
            for j in range(rng.integers(5, 9)):
                new.append([r.user_id, r.ts + pd.Timedelta(seconds=40*j + rng.uniform(0, 20)),
                            u.mean_amt*rng.uniform(.8, 1.5), r.channel, r.city, 1, "burst"])
    if new: df = pd.concat([df, pd.DataFrame(new, columns=df.columns)], ignore_index=True)
    return df.sort_values(["ts"]).reset_index(drop=True)

def build_features(df, users):
    d = df.merge(users[["user_id","home","mean_amt"]].rename(columns={"home":"home_city"}), on="user_id")
    d = d.sort_values(["user_id","ts"]).reset_index(drop=True)
    g = d.groupby("user_id")
    d["gap_s"] = g.ts.diff().dt.total_seconds().fillna(86400).clip(lower=1)
    d["lat"] = d.city.map(lambda c: CITIES[c][0]); d["lon"] = d.city.map(lambda c: CITIES[c][1])
    d["dist_km"] = haversine(g.lat.shift().fillna(d.lat), g.lon.shift().fillna(d.lon), d.lat, d.lon)
    hl = d.home_city.map(lambda c: CITIES[c][0]); ho = d.home_city.map(lambda c: CITIES[c][1])
    d["dist_home_km"] = haversine(hl, ho, d.lat, d.lon)
    d["speed_kmh"] = d.dist_km / (d.gap_s/3600)
    d["log_amount"] = np.log1p(d.amount)
    d["amount_vs_user"] = np.log(d.amount / d.mean_amt)
    d["log_gap_s"] = np.log(d.gap_s)
    hr = d.ts.dt.hour + d.ts.dt.minute/60
    d["hour_sin"], d["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    d["speed_kmh_log"] = np.log1p(d.speed_kmh)
    d["is_upi"] = (d.channel == "UPI").astype(int)
    # frequency: txns by same user in the trailing 1 hour (incl. current)
    cnt = np.ones(len(d))
    for _, idx in d.groupby("user_id").indices.items():
        t = d.ts.values[idx].astype("int64")//10**9
        cnt[idx] = np.arange(len(t)) - np.searchsorted(t, t-3600, side="left") + 1
    d["txn_count_1h"] = cnt
    return d.sort_values("ts").reset_index(drop=True)

class Detector:
    """Isolation Forest + autoencoder"""
    def __init__(self, alert_rate=0.01, seed=42):
        self.alert_rate, self.seed = alert_rate, seed
    def fit(self, X):
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        self.iso = IsolationForest(n_estimators=300, random_state=self.seed).fit(Z)
        self.ae = MLPRegressor(hidden_layer_sizes=(32, 6, 32), activation="tanh", max_iter=800,
                               learning_rate_init=3e-3, random_state=self.seed).fit(Z, Z)
        s = self._raw(Z); self.ref = {k: np.sort(v) for k, v in s.items()}
        self.threshold = np.quantile(self.score(X)["score"], 1 - self.alert_rate)
        return self
    def _raw(self, Z):
        err = (Z - self.ae.predict(Z))**2
        return {"iso": -self.iso.score_samples(Z), "ae": err.mean(1), "ae_feat": err}
    def score(self, X):
        Z = self.sc.transform(X); r = self._raw(Z)
        pct = lambda k: np.searchsorted(self.ref[k], r[k]) / len(self.ref[k])   # percentile vs training
        return {"score": (pct("iso") + pct("ae"))/2, "iso": r["iso"], "ae": r["ae"], "ae_feat": r["ae_feat"]}

def explain(row):
    """Human-readable reasons"""
    r = []
    if row.amount_vs_user > 1.6: r.append(f"amount ~{np.exp(row.amount_vs_user):.0f}x user's usual")
    if row.speed_kmh > 400: r.append(f"impossible travel ({row.dist_km:.0f} km in {row.gap_s/60:.0f} min)")
    if row.txn_count_1h >= 4: r.append(f"{int(row.txn_count_1h)} txns in last hour")
    if row.ts.hour < 5: r.append(f"unusual hour ({row.ts.strftime('%H:%M')})")
    if row.dist_home_km > 500 and row.speed_kmh <= 400: r.append(f"far from home city ({row.city})")
    return r or ["unusual combination of features"]
