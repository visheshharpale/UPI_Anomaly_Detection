"""Core logic: synthetic data, feature engineering, models, scoring.
Shared by main.py and the Flask app so both use identical code.
Vectorised so it scales to 1M+ transactions."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CITIES = {  # name: (lat, lon)
    "Mumbai": (19.07, 72.88), "Pune": (18.52, 73.86), "Delhi": (28.61, 77.21),
    "Bengaluru": (12.97, 77.59), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36),
    "Hyderabad": (17.39, 78.49), "Ahmedabad": (23.02, 72.57)}
CITY_NAMES = list(CITIES)
LAT = {c: v[0] for c, v in CITIES.items()}
LON = {c: v[1] for c, v in CITIES.items()}
FEATURES = ["log_amount", "amount_vs_user", "log_gap_s", "hour_sin", "hour_cos",
            "dist_km", "speed_kmh_log", "dist_home_km", "txn_count_1h", "is_upi"]
HOUR_NS = 3.6e12   # nanoseconds in one hour

def _td(x, unit_ns):
    """float array -> timedelta64[ns] array (unit_ns = nanoseconds per unit)."""
    return (np.asarray(x) * unit_ns).astype("int64").astype("timedelta64[ns]")

def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 12742 * np.arcsin(np.sqrt(a))

def make_users(n=12000, rng=None):
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame({
        "user_id": np.arange(n),
        "home": rng.choice(CITY_NAMES, n, p=[.3,.15,.15,.12,.08,.07,.07,.06]),
        "mean_amt": rng.lognormal(6.3, .6, n),       # typical ticket size (Rs)
        "peak_hour": rng.normal(15, 3, n).clip(9, 21),
        "txn_per_day": rng.uniform(1, 5, n),
        "upi_share": rng.uniform(.3, .95, n)})

def generate(users, start, days=30, anomaly_rate=0.01, anom_after=None, seed=0):
    """Simulate normal behaviour + inject 4 fraud patterns (fully vectorised).
    'label'/'kind' are kept ONLY for demo/sanity checks, never for training."""
    rng = np.random.default_rng(seed)
    to_ns = lambda t: pd.Timestamp(t).to_datetime64().astype("datetime64[ns]")
    start = to_ns(start); anom_after = start if anom_after is None else to_ns(anom_after)
    U = len(users)
    uid_arr, mean_amt = users.user_id.to_numpy(), users.mean_amt.to_numpy()
    home = users.home.to_numpy(dtype=object)
    home_idx = np.array([CITY_NAMES.index(h) for h in home])

    # ---- normal behaviour: one Poisson count per (user, day), then expand ----
    counts = rng.poisson(np.repeat(users.txn_per_day.to_numpy(), days))
    uidx = np.repeat(np.repeat(np.arange(U), days), counts)       # which user each txn belongs to
    day = np.repeat(np.tile(np.arange(days), U), counts)
    n = len(uidx)
    hour = np.clip(rng.normal(users.peak_hour.to_numpy()[uidx], 3), 6, 23.5)
    ts = start + _td(day*24 + hour, HOUR_NS)
    amount = rng.lognormal(np.log(mean_amt[uidx]), .5)
    channel = np.where(rng.random(n) < users.upi_share.to_numpy()[uidx], "UPI", "ATM").astype(object)
    city = home[uidx].copy()
    label = np.zeros(n, dtype=int); kind = np.full(n, "normal", dtype=object)

    # ---- fraud injection ----
    pool = np.where(ts >= anom_after)[0]
    k = max(1, int(anomaly_rate*len(pool))) if anomaly_rate > 0 else 0
    chosen = rng.choice(pool, k, replace=False) if k else np.array([], dtype=int)
    ktype = rng.integers(0, 4, len(chosen))

    i = chosen[ktype == 0]                                         # huge amount
    amount[i] = mean_amt[uidx[i]] * rng.uniform(12, 40, len(i)); label[i] = 1; kind[i] = "big_amount"

    i = chosen[ktype == 1]                                         # 2-4 am, large amount
    ts[i] = ts[i].astype("datetime64[D]").astype("datetime64[ns]") + _td(rng.uniform(2, 4, len(i)), HOUR_NS)
    amount[i] = mean_amt[uidx[i]] * rng.uniform(4, 9, len(i)); label[i] = 1; kind[i] = "odd_hour"

    i = chosen[ktype == 2]                                         # far-away city minutes later
    city[i] = np.array(CITY_NAMES, dtype=object)[(home_idx[uidx[i]] + rng.integers(1, len(CITY_NAMES), len(i))) % len(CITY_NAMES)]
    ts[i] = ts[i] + _td(rng.uniform(5, 40, len(i)), 60e9); label[i] = 1; kind[i] = "location_jump"

    df = pd.DataFrame({"user_id": uid_arr[uidx], "ts": ts, "amount": amount, "channel": channel,
                       "city": city, "label": label, "kind": kind})

    i = chosen[ktype == 3]                                         # burst: 5-8 extra rapid txns
    if len(i):
        nb = rng.integers(5, 9, len(i)); tot = nb.sum(); rep = np.repeat(i, nb)
        j = np.arange(tot) - np.repeat(np.cumsum(nb) - nb, nb)     # 0..nb-1 within each burst
        burst = pd.DataFrame({"user_id": uid_arr[uidx[rep]],
            "ts": ts[rep] + _td(40*j + rng.uniform(0, 20, tot), 1e9),
            "amount": mean_amt[uidx[rep]] * rng.uniform(.8, 1.5, tot),
            "channel": channel[rep], "city": city[rep], "label": 1, "kind": "burst"})
        df = pd.concat([df, burst], ignore_index=True)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)

def build_features(df, users):
    d = df.merge(users[["user_id","home","mean_amt"]].rename(columns={"home":"home_city"}), on="user_id")
    d = d.sort_values(["user_id","ts"]).reset_index(drop=True)
    d["lat"] = d.city.map(LAT).astype(float); d["lon"] = d.city.map(LON).astype(float)
    g = d.groupby("user_id")
    d["gap_s"] = g.ts.diff().dt.total_seconds().fillna(86400).clip(lower=1)
    d["dist_km"] = haversine(g.lat.shift().fillna(d.lat), g.lon.shift().fillna(d.lon), d.lat, d.lon)
    d["dist_home_km"] = haversine(d.home_city.map(LAT).astype(float), d.home_city.map(LON).astype(float), d.lat, d.lon)
    d["speed_kmh"] = d.dist_km / (d.gap_s/3600)
    d["log_amount"] = np.log1p(d.amount)
    d["amount_vs_user"] = np.log(d.amount / d.mean_amt)
    d["log_gap_s"] = np.log(d.gap_s)
    hr = d.ts.dt.hour + d.ts.dt.minute/60
    d["hour_sin"], d["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    d["speed_kmh_log"] = np.log1p(d.speed_kmh)
    d["is_upi"] = (d.channel == "UPI").astype(int)
    # frequency: txns by the same user in the trailing hour (incl. current), fully vectorised.
    # Rows are sorted by (user, time), so key = user*1e10 + epoch_seconds is globally increasing.
    sec = d.ts.to_numpy().astype("datetime64[s]").astype("int64")
    key = d.user_id.to_numpy().astype("int64") * 10**10 + sec
    d["txn_count_1h"] = np.arange(len(d)) - np.searchsorted(key, key - 3600, side="left") + 1
    return d.sort_values("ts", kind="stable").reset_index(drop=True)

class Detector:
    """Isolation Forest + autoencoder. Trains on a random sample (max_train rows), scores any amount."""
    def __init__(self, alert_rate=0.01, seed=42, max_train=200_000):
        self.alert_rate, self.seed, self.max_train = alert_rate, seed, max_train
    def fit(self, X):
        if len(X) > self.max_train:        # no need to train on all 1M rows
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)]
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(Z)
        self.ae = MLPRegressor(hidden_layer_sizes=(32, 6, 32), activation="tanh", max_iter=150,
                               batch_size=1024, learning_rate_init=3e-3, n_iter_no_change=10,
                               random_state=self.seed).fit(Z, Z)
        s = self._raw(Z); self.ref = {k: np.sort(v) for k, v in s.items() if k != "ae_feat"}
        self.threshold = np.quantile(self.score(X)["score"], 1 - self.alert_rate)
        return self
    def _raw(self, Z):
        err = (Z - self.ae.predict(Z))**2
        return {"iso": -self.iso.score_samples(Z), "ae": err.mean(1), "ae_feat": err}
    def score(self, X, batch=250_000):
        out = {"score": [], "iso": [], "ae": [], "ae_feat": []}
        for a in range(0, len(X), batch):          # batches keep memory flat for 1M+ rows
            r = self._raw(self.sc.transform(X[a:a+batch]))
            pct = lambda k: np.searchsorted(self.ref[k], r[k]) / len(self.ref[k])   # percentile vs training
            out["score"].append((pct("iso") + pct("ae"))/2)
            for k in ("iso", "ae", "ae_feat"): out[k].append(r[k])
        return {k: np.concatenate(v) for k, v in out.items()}

def explain(row):
    """Human-readable reasons"""
    r = []
    if row.amount_vs_user > 1.6: r.append(f"amount ~{np.exp(row.amount_vs_user):.0f}x user's usual")
    if row.speed_kmh > 400: r.append(f"impossible travel ({row.dist_km:.0f} km in {row.gap_s/60:.0f} min)")
    if row.txn_count_1h >= 4: r.append(f"{int(row.txn_count_1h)} txns in last hour")
    if row.ts.hour < 5: r.append(f"unusual hour ({row.ts.strftime('%H:%M')})")
    if row.dist_home_km > 500 and row.speed_kmh <= 400: r.append(f"far from home city ({row.city})")
    return r or ["unusual combination of features"]