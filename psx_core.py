# =====================================================================
#  psx_core.py  -  shared logic for PSX PulseCast (training + GUI)
#  Features, ensemble model, recursive multi-day forecast, buy/sell signal
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIELDS = ["Open", "High", "Low", "Close"]
TARGETS = ["ret_open", "ret_high", "ret_low", "ret_close"]
MIN_HISTORY = 40            # minimum past trading days needed before a forecast start date
MA_WINDOWS = [5, 9, 21, 200]
WEIGHTS = {"hgb": 0.5, "et": 0.3, "ridge": 0.2}
# Open/Close daily moves are close to random noise -> shrink them toward 0 (found on time-based hold-outs);
# High/Low (day range) carry real signal -> kept at full strength.
SHRINK = {"ret_open": 0.4, "ret_high": 1.0, "ret_low": 1.0, "ret_close": 0.4}
FEATURE_TAIL = 260          # rows used when rebuilding features during forecasting


# ---------------------------------------------------------------- features
def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n, min_periods=3).mean()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=3).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def make_features(df):
    """df = ONE company, sorted by Date, columns Open High Low Close."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=df.index)
    for k in range(1, 6):
        f[f"ret_lag{k}"] = r.shift(k - 1)
    f["gap"] = np.log(o / c.shift(1))
    f["body"] = np.log(c / o)
    f["range"] = np.log(h / l)
    f["range_lag1"] = f["range"].shift(1)
    f["upper"] = np.log(h / np.maximum(o, c))
    f["lower"] = np.log(np.minimum(o, c) / l)
    span = (h - l).replace(0, np.nan)
    f["clv"] = ((c - l) / span).fillna(0.5)
    for w in (5, 9, 21):
        f[f"dist_ma{w}"] = np.log(c / c.rolling(w, min_periods=w).mean())
    f["dist_ma200"] = np.log(c / c.rolling(200, min_periods=30).mean())
    f["ma9_21"] = np.log(c.rolling(9, min_periods=9).mean() / c.rolling(21, min_periods=21).mean())
    f["vol5"] = r.rolling(5, min_periods=3).std()
    f["vol21"] = r.rolling(21, min_periods=5).std()
    f["rsi14"] = _rsi(c) / 100 - 0.5
    for k in (5, 10, 21):
        f[f"mom{k}"] = lc - lc.shift(k)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_training(df_all):
    """Pooled supervised table: features at day t -> next-day OHLC log-returns vs Close(t)."""
    parts = []
    for sym, g in df_all.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        f = make_features(g)
        f["dow_next"] = g["Date"].shift(-1).dt.dayofweek
        c = g["Close"]
        f["ret_open"] = np.log(g["Open"].shift(-1) / c)
        f["ret_high"] = np.log(g["High"].shift(-1) / c)
        f["ret_low"] = np.log(g["Low"].shift(-1) / c)
        f["ret_close"] = np.log(g["Close"].shift(-1) / c)
        f["Symbol"] = sym
        f["Date"] = g["Date"]                       # date of day t (features)
        f["NextDate"] = g["Date"].shift(-1)
        f["PrevClose"] = c
        f["NextOpen"] = g["Open"].shift(-1)
        f["NextHigh"] = g["High"].shift(-1)
        f["NextLow"] = g["Low"].shift(-1)
        f["NextClose"] = g["Close"].shift(-1)
        parts.append(f.iloc[21:].dropna(subset=TARGETS + ["dow_next"]))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["Date", "Symbol"]).reset_index(drop=True)


def feature_columns(train_df):
    drop = set(TARGETS + ["Symbol", "Date", "NextDate", "PrevClose",
                          "NextOpen", "NextHigh", "NextLow", "NextClose"])
    return [c for c in train_df.columns if c not in drop]


# ------------------------------------------------------------------- model
def fit_ensemble(X, Y, hgb_params=None, seed=42):
    """One ensemble (HistGradientBoosting + ExtraTrees + Ridge) per target."""
    hp = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=60,
              l2_regularization=5.0, random_state=seed)
    if hgb_params:
        hp.update(hgb_params)
    models = {}
    for t in TARGETS:
        y = Y[t].values
        models[t] = {
            "hgb": HistGradientBoostingRegressor(**hp).fit(X, y),
            "et": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=25, max_features=0.5,
                                      n_jobs=-1, random_state=seed).fit(X, y),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=200.0)).fit(X, y),
        }
    return models


def predict_returns(models, X):
    out = {}
    for t in TARGETS:
        m = models[t]
        out[t] = SHRINK[t] * (WEIGHTS["hgb"] * m["hgb"].predict(X)
                              + WEIGHTS["et"] * m["et"].predict(X)
                              + WEIGHTS["ridge"] * m["ridge"].predict(X))
    return out


# ------------------------------------------------------------ date helpers
def plan_dates(dates, start, n):
    """Return (cut, hidden_dates, shown_dates).
    cut    = number of history rows that are strictly BEFORE the start date
    hidden = business days that must be predicted first (only if start is beyond the data)
    shown  = the n dates the user wants predictions for"""
    dates = pd.DatetimeIndex(dates)
    last = dates[-1]
    start = pd.Timestamp(start).normalize()
    cut = int((dates < start).sum())
    if cut < MIN_HISTORY:
        raise ValueError(f"Choose a start date after {dates[MIN_HISTORY - 1].date()} "
                         f"(the model needs {MIN_HISTORY} days of history).")
    hidden = []
    if start <= last:
        shown = list(dates[cut:cut + n])
    else:
        gap_end = start - pd.Timedelta(days=1)
        if gap_end > last:
            hidden = list(pd.bdate_range(last + pd.Timedelta(days=1), gap_end))
        shown = list(pd.bdate_range(start, periods=n))
    if len(shown) < n:                      # run past the last known date -> continue on business days
        anchor = shown[-1] if shown else last
        shown += list(pd.bdate_range(anchor + pd.Timedelta(days=1), periods=n - len(shown)))
    return cut, hidden, shown


# ----------------------------------------------------------------- signals
def make_signal(pred_ret, vol, ma5, ma21, rsi):
    """BUY / SELL / HOLD from predicted move, short-term trend and RSI."""
    vol = vol if vol and vol > 1e-4 else 0.015
    ml = float(np.clip(pred_ret / (0.5 * vol), -1, 1))
    trend = 0.4 if ma5 > ma21 else -0.4
    adj = -0.2 if rsi > 75 else (0.2 if rsi < 25 else 0.0)
    score = ml + trend + adj
    if score >= 0.3:
        return "BUY", score
    if score <= -0.3:
        return "SELL", score
    return "HOLD", score


# --------------------------------------------------------------- forecasting
def forecast(history, models, feat_cols, target_dates, show_from=0):
    """Recursive multi-day OHLC forecast for ONE company.
    history      : DataFrame Date/Open/High/Low/Close (only data known at the forecast origin)
    target_dates : consecutive trading dates to predict (hidden + shown)
    show_from    : index in target_dates where the rows the user asked for start"""
    h = history[["Date"] + FIELDS].sort_values("Date").reset_index(drop=True)
    rows = []
    for d in target_dates:
        f = make_features(h.tail(FEATURE_TAIL)).iloc[[-1]].copy()
        f["dow_next"] = pd.Timestamp(d).dayofweek
        X = f[feat_cols]
        r = predict_returns(models, X)
        pc = float(h["Close"].iloc[-1])
        o = pc * float(np.exp(r["ret_open"][0]))
        hi = pc * float(np.exp(r["ret_high"][0]))
        lo = pc * float(np.exp(r["ret_low"][0]))
        c = pc * float(np.exp(r["ret_close"][0]))
        hi = max(hi, o, c)
        lo = min(lo, o, c)
        new = {"Date": pd.Timestamp(d), "Open": o, "High": hi, "Low": lo, "Close": c}
        h = pd.concat([h, pd.DataFrame([new])], ignore_index=True)
        closes = h["Close"]
        sig, score = make_signal(np.log(c / pc), float(f["vol21"].iloc[0]),
                                 closes.tail(5).mean(), closes.tail(21).mean(),
                                 float(_rsi(closes).iloc[-1]))
        new.update({"Prev Close": pc, "Expected Change %": (c / pc - 1) * 100,
                    "Signal": sig, "Score": score})
        rows.append(new)
    out = pd.DataFrame(rows).iloc[show_from:].reset_index(drop=True)
    return out


# -------------------------------------------------- history for the charts
def add_indicators(df):
    """Adds MA5/9/21/200 and bullish/bearish regime (MA9 vs MA21) to one company's history."""
    d = df.sort_values("Date").reset_index(drop=True).copy()
    for w in MA_WINDOWS:
        d[f"MA{w}"] = d["Close"].rolling(w, min_periods=w).mean()
    d["Trend"] = np.where(d["MA9"].isna() | d["MA21"].isna(), "",
                          np.where(d["MA9"] >= d["MA21"], "Bullish", "Bearish"))
    return d
