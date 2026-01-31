# BTC Strategy Trade Visualiser (v4.9)
# Upgrades included:
# - Multi-strategy engine
# - Strategy-specific exits (AUTO + override)
# - Confluence scoring (weights + min score filter)
# - Walk-forward ML (rolling train/test windows; no leakage)
# - Session overlays (Asia/London/NY boxes)
# - Realistic entry: signal on candle close -> fill next candle open (default ON)
# - Intrabar TP/SL tie-break: worst/best
# - ML P(win) overlay on chart at signal points
# - Beginner Mode: simple + detailed explanations + what changing params does

import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BTC Strategy Trade Visualiser (v4.9)", layout="wide")

# =====================
# LOADER
# =====================
def _read_binance_like_csv(fileobj):
    df = pd.read_csv(fileobj, header=None)
    if df.shape[1] < 6:
        return pd.DataFrame()
    df = df.iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    return df


def load_files(files):
    dfs = []
    for f in files:
        raw = f.read()
        name = f.name.lower()

        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for n in z.namelist():
                    if n.lower().endswith(".csv"):
                        try:
                            with z.open(n) as fh:
                                dfs.append(_read_binance_like_csv(fh))
                        except Exception:
                            continue
        elif name.endswith(".csv"):
            dfs.append(_read_binance_like_csv(io.BytesIO(raw)))

    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])

    # robust timestamp unit detection
    t = float(df["open_time"].median())
    if t > 1e17:
        unit = "ns"
    elif t > 1e14:
        unit = "us"
    elif t > 1e11:
        unit = "ms"
    else:
        unit = "s"

    dt_utc = pd.to_datetime(df["open_time"].astype("int64"), unit=unit, utc=True, errors="coerce")
    df["dt"] = dt_utc.dt.tz_convert("UTC").dt.tz_localize(None)

    df = df.dropna(subset=["dt"])
    df = df[(df["dt"] >= pd.Timestamp("2017-01-01")) & (df["dt"] <= pd.Timestamp("2035-12-31"))]
    df = df.sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_files_cached(file_payload):
    class F:
        def __init__(self, name, b):
            self.name = name
            self._b = b

        def read(self):
            return self._b

    files = [F(n, b) for n, b in file_payload]
    return load_files(files)


# =====================
# TIMEFRAME / RESAMPLE
# =====================
def infer_base_minutes(df: pd.DataFrame) -> float | None:
    if df.empty or len(df) < 3:
        return None
    diffs = df["dt"].diff().dropna()
    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return None
    return med.total_seconds() / 60.0


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("dt").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = d.resample(rule, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


@st.cache_data(show_spinner=False)
def resample_cached(df_view: pd.DataFrame, rule: str) -> pd.DataFrame:
    return resample_ohlcv(df_view, rule)


# =====================
# INDICATORS
# =====================
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def roc(s, n):
    return s.pct_change(n) * 100.0


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    width_pct = (upper - lower) / close * 100.0
    return mid, upper, lower, width_pct


def vwap_session(df):
    d = df.copy()
    day = d["dt"].dt.floor("D")
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    pv = tp * d["volume"]
    return pv.groupby(day).cumsum() / d["volume"].groupby(day).cumsum().replace(0, np.nan)


def vwap_anchored_full(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)


# =====================
# CONFLUENCE SCORING
# =====================
def normalize_score(parts: dict[str, float], weights: dict[str, float]) -> float:
    # score = weighted average of components in [0,1] -> [0,100]
    wsum = 0.0
    ssum = 0.0
    for k, v in parts.items():
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        v = float(np.clip(v, 0.0, 1.0))
        ssum += w * v
        wsum += w
    return float((ssum / wsum) * 100.0) if wsum > 0 else 0.0


# =====================
# STRATEGY ENGINE
# =====================
STRATEGIES = [
    "MA Crossover",
    "Momentum (ROC+VOL+RSI)",
    "Trend Following (EMA pullback)",
    "Range Breakout",
    "Bollinger Squeeze Breakout",
    "RSI Mean Reversion",
    "VWAP Reversion",
    "Liquidity Sweep (basic)",
]

EXIT_MODELS = ["AUTO", "TP/SL only", "ATR Trailing Stop", "EMA Flip Exit", "Time Stop", "Opposite Signal"]
DEFAULT_EXIT_FOR_STRAT = {
    "MA Crossover": "Opposite Signal",
    "Momentum (ROC+VOL+RSI)": "ATR Trailing Stop",
    "Trend Following (EMA pullback)": "EMA Flip Exit",
    "Range Breakout": "TP/SL only",
    "Bollinger Squeeze Breakout": "TP/SL only",
    "RSI Mean Reversion": "Time Stop",
    "VWAP Reversion": "Time Stop",
    "Liquidity Sweep (basic)": "TP/SL only",
}


def compute_signals(
    d: pd.DataFrame,
    strategy_name: str,
    fast: int,
    slow: int,
    trend_len: int,
    rsi_len: int,
    roc_len: int,
    vol_len: int,
    range_len: int,
    break_close: bool,
    bb_len: int,
    bb_k: float,
    squeeze_pct: float,
    use_session_vwap: bool,
    vwap_dev: float,
    momentum_roc_thresh: float,
    weights: dict[str, float],
    min_score: float,
):
    x = d.copy()

    # shared indicators
    x["RSI"] = rsi(x["close"], rsi_len)
    x["ROC"] = roc(x["close"], roc_len)
    x["VOL_MA"] = x["volume"].rolling(vol_len).mean()
    x["VOL_EXP"] = x["volume"] > (x["VOL_MA"] * 1.3)

    x["long_sig"] = False
    x["short_sig"] = False
    x["sig_tag"] = ""
    x["score"] = 0.0

    def apply_score(mask_long: pd.Series, mask_short: pd.Series, parts_df: pd.DataFrame, tag_long: str, tag_short: str):
        parts_keys = list(parts_df.columns)
        for k in parts_keys:
            x[f"score_{k}"] = parts_df[k].astype(float)

        scores = []
        for i in range(len(x)):
            parts = {k: float(parts_df.iloc[i][k]) for k in parts_keys}
            scores.append(normalize_score(parts, weights))
        x["score"] = np.array(scores, dtype=float)

        ok = x["score"] >= float(min_score)
        x["long_sig"] = mask_long & ok
        x["short_sig"] = mask_short & ok
        x.loc[x["long_sig"], "sig_tag"] = tag_long
        x.loc[x["short_sig"], "sig_tag"] = tag_short

    if strategy_name == "MA Crossover":
        x["FAST"] = sma(x["close"], fast)
        x["SLOW"] = sma(x["close"], slow)
        x["TREND"] = sma(x["close"], trend_len)

        prev_fast = x["FAST"].shift(1)
        prev_slow = x["SLOW"].shift(1)
        cross_up = (prev_fast <= prev_slow) & (x["FAST"] > x["SLOW"])
        cross_dn = (prev_fast >= prev_slow) & (x["FAST"] < x["SLOW"])

        parts = pd.DataFrame(
            {
                "trend": ((x["close"] > x["TREND"]).astype(float)),
                "rsi": ((x["RSI"] - 50.0) / 50.0).clip(0, 1),
                "ma_sep": ((x["FAST"] - x["SLOW"]) / x["close"]).abs().clip(0, 0.01) / 0.01,
            }
        )
        apply_score(cross_up, cross_dn, parts, "MA_CROSS_UP", "MA_CROSS_DN")

    elif strategy_name == "Momentum (ROC+VOL+RSI)":
        Xthr = float(momentum_roc_thresh)
        long_raw = (x["ROC"] > Xthr) & (x["RSI"] > 60) & (x["VOL_EXP"])
        short_raw = (x["ROC"] < -Xthr) & (x["RSI"] < 40) & (x["VOL_EXP"])

        parts = pd.DataFrame(
            {
                "roc": (x["ROC"].abs() / max(1e-9, Xthr)).clip(0, 2) / 2,
                "vol": x["VOL_EXP"].astype(float),
                "rsi": ((x["RSI"] - 50.0).abs() / 50.0).clip(0, 1),
            }
        )
        apply_score(long_raw, short_raw, parts, "MOMO_LONG", "MOMO_SHORT")

    elif strategy_name == "Trend Following (EMA pullback)":
        x["EMA20"] = ema(x["close"], 20)
        x["EMA50"] = ema(x["close"], 50)
        x["ema_slope"] = (x["EMA50"] - x["EMA50"].shift(10)) / x["close"]

        trend_up = (x["EMA20"] > x["EMA50"]) & (x["ema_slope"] > 0)
        trend_dn = (x["EMA20"] < x["EMA50"]) & (x["ema_slope"] < 0)

        pullback = (x["close"] >= x["EMA20"] * 0.995) & (x["close"] <= x["EMA20"] * 1.005)

        long_raw = trend_up & pullback & (x["RSI"] > 55)
        short_raw = trend_dn & pullback & (x["RSI"] < 45)

        parts = pd.DataFrame(
            {
                "trend": ((x["EMA20"] - x["EMA50"]) / x["close"]).clip(-0.01, 0.01).abs() / 0.01,
                "pullback": pullback.astype(float),
                "rsi": ((x["RSI"] - 50.0).abs() / 50.0).clip(0, 1),
            }
        )
        apply_score(long_raw, short_raw, parts, "TREND_PB_LONG", "TREND_PB_SHORT")

    elif strategy_name == "Range Breakout":
        x["R_H"] = x["high"].rolling(range_len).max()
        x["R_L"] = x["low"].rolling(range_len).min()

        if break_close:
            long_raw = x["close"] > x["R_H"].shift(1)
            short_raw = x["close"] < x["R_L"].shift(1)
        else:
            long_raw = x["high"] > x["R_H"].shift(1)
            short_raw = x["low"] < x["R_L"].shift(1)

        x["ATR14"] = atr(x, 14)
        atr_pct = (x["ATR14"] / x["close"]).replace([np.inf, -np.inf], np.nan)

        dist_up = ((x["close"] - x["R_H"].shift(1)) / x["close"]).clip(0, 0.01) / 0.01
        dist_dn = ((x["R_L"].shift(1) - x["close"]) / x["close"]).clip(0, 0.01) / 0.01

        parts = pd.DataFrame(
            {
                "compression": (1.0 - (atr_pct / atr_pct.rolling(200).median())).clip(0, 1).fillna(0),
                "strength": np.maximum(dist_up, dist_dn).fillna(0),
                "vol": x["VOL_EXP"].astype(float),
            }
        )
        apply_score(long_raw, short_raw, parts, "RANGE_BO_LONG", "RANGE_BO_SHORT")

    elif strategy_name == "Bollinger Squeeze Breakout":
        mid, up, lo, width = bollinger(x["close"], bb_len, bb_k)
        x["BB_M"] = mid
        x["BB_U"] = up
        x["BB_L"] = lo
        x["BB_W"] = width

        squeeze = x["BB_W"] < float(squeeze_pct)
        long_raw = squeeze.shift(1) & (x["close"] > x["BB_U"])
        short_raw = squeeze.shift(1) & (x["close"] < x["BB_L"])

        parts = pd.DataFrame(
            {
                "compression": (1.0 - (x["BB_W"] / float(squeeze_pct)).clip(0, 1)).fillna(0),
                "strength": ((x["close"] - x["BB_M"]).abs() / x["close"]).clip(0, 0.01) / 0.01,
                "vol": x["VOL_EXP"].astype(float),
            }
        )
        apply_score(long_raw, short_raw, parts, "BB_SQZ_LONG", "BB_SQZ_SHORT")

    elif strategy_name == "RSI Mean Reversion":
        long_raw = x["RSI"] < 30
        short_raw = x["RSI"] > 70

        parts = pd.DataFrame(
            {
                "rsi": ((50.0 - x["RSI"]).clip(0, 50) / 50.0).fillna(0).where(
                    long_raw, ((x["RSI"] - 50.0).clip(0, 50) / 50.0).fillna(0)
                ),
                "vol": (1.0 - x["VOL_EXP"].astype(float)),
                "roc": (x["ROC"].abs().clip(0, 2) / 2).fillna(0),
            }
        )
        apply_score(long_raw, short_raw, parts, "RSI_MR_LONG", "RSI_MR_SHORT")

    elif strategy_name == "VWAP Reversion":
        x["VWAP"] = vwap_session(x) if use_session_vwap else vwap_anchored_full(x)
        dev = (x["close"] - x["VWAP"]) / x["VWAP"] * 100.0
        x["VWAP_DEV"] = dev

        thr = abs(float(vwap_dev))
        long_raw = dev < -thr
        short_raw = dev > thr

        parts = pd.DataFrame(
            {
                "dev": (dev.abs() / thr).clip(0, 2) / 2,
                "vol": (1.0 - x["VOL_EXP"].astype(float)),
                "rsi": ((x["RSI"] - 50.0).abs() / 50.0).clip(0, 1),
            }
        )
        apply_score(long_raw, short_raw, parts, "VWAP_REV_LONG", "VWAP_REV_SHORT")

    elif strategy_name == "Liquidity Sweep (basic)":
        look = 20
        eq_tol = 0.0008
        hh = x["high"].rolling(look).max()
        ll = x["low"].rolling(look).min()
        eq_high = hh.shift(1)
        eq_low = ll.shift(1)

        sweep_high = (x["high"] > eq_high * (1 + eq_tol)) & (x["close"] < eq_high)
        sweep_low = (x["low"] < eq_low * (1 - eq_tol)) & (x["close"] > eq_low)

        long_raw = sweep_low
        short_raw = sweep_high

        wick_up = ((x["high"] - x[["open", "close"]].max(axis=1)) / x["close"]).clip(0, 0.02) / 0.02
        wick_dn = ((x[["open", "close"]].min(axis=1) - x["low"]) / x["close"]).clip(0, 0.02) / 0.02

        parts = pd.DataFrame(
            {
                "wick": np.maximum(wick_up, wick_dn).fillna(0),
                "reclaim": ((x["close"] - (eq_high.fillna(x["close"]))) / x["close"]).abs().clip(0, 0.01) / 0.01,
                "vol": x["VOL_EXP"].astype(float),
            }
        )
        apply_score(long_raw, short_raw, parts, "LIQ_SWEEP_LONG", "LIQ_SWEEP_SHORT")

    return x


# =====================
# ML: Logistic Regression (no sklearn)
# =====================
def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X, y, lr=0.2, steps=700, l2=1e-2, seed=7):
    rng = np.random.default_rng(seed)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd

    mask = np.isfinite(Xs).all(axis=1) & np.isfinite(y)
    Xs = Xs[mask]
    y = y[mask].astype(float)

    if len(y) < 150:
        return None

    w = rng.normal(0, 0.01, size=Xs.shape[1])
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(Xs @ w + b)
        err = p - y
        gw = (Xs.T @ err) / len(y) + l2 * w
        gb = np.mean(err)
        w -= lr * gw
        b -= lr * gb

    return w, b, mu, sd


def predict_logreg_proba(X, model):
    w, b, mu, sd = model
    Xs = (X - mu) / sd
    p = np.full(X.shape[0], np.nan, dtype=float)
    mask = np.isfinite(Xs).all(axis=1)
    p[mask] = _sigmoid(Xs[mask] @ w + b)
    return p


def build_ml_features(df: pd.DataFrame):
    d = df.copy()
    d["ATR14"] = atr(d, 14)
    ret = d["close"].pct_change()
    d["ret_std50"] = ret.rolling(50).std()
    d["slope10"] = (d["close"] - d["close"].shift(10)) / d["close"]
    d["body_pct"] = (d["close"] - d["open"]).abs() / d["close"]
    d["range_pct"] = (d["high"] - d["low"]) / d["close"]
    d["atr_pct"] = d["ATR14"] / d["close"]

    d["score_n"] = (d["score"] / 100.0).clip(0, 1) if "score" in d.columns else 0.0
    d["rsi_n"] = (d["RSI"] / 100.0).clip(0, 1) if "RSI" in d.columns else np.nan
    d["roc_n"] = (d["ROC"].clip(-5, 5) + 5) / 10.0 if "ROC" in d.columns else np.nan

    cols = ["atr_pct", "ret_std50", "slope10", "body_pct", "range_pct", "score_n", "rsi_n", "roc_n"]
    X = d[cols].to_numpy(dtype=float)
    return X, cols


# =====================
# BACKTEST
# =====================
def backtest(
    df,
    atr_len,
    atr_mult,
    rr,
    risk,
    both_dirs,
    exit_on_opp,
    fill_next_open=True,
    intrabar_rule="Worst case (SL first)",
    exit_model="TP/SL only",
    trail_atr_mult=2.0,
    time_stop_bars=48,
    use_ml=False,
    proba=None,
    proba_thresh=0.55,
):
    # Defensive checks (prevents "NoneType has no attribute copy" crashes)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"backtest expected a pandas DataFrame, got {type(df)}. "
            "This usually means compute_signals returned None or a cached result is corrupted. "
            "Click 'Clear cache' in the sidebar and rerun, and ensure you deployed the correct app file."
        )
    if df.empty:
        return df, pd.DataFrame()

    d = df.copy()
    d["ATR"] = atr(d, atr_len)

    trades = []
    pos = 0
    entry = sl = tp = None
    entry_time = None
    signal_time = None
    signal_proba = None
    signal_tag = None
    bars_in_trade = 0
    trail = None

    start_i = max(atr_len, 5) + 2
    end_limit = len(d) - 2 if fill_next_open else len(d) - 1
    if len(d) <= start_i + 2:
        return d, pd.DataFrame()

    intrabar_is_best = intrabar_rule.lower().startswith("best")

    for i in range(start_i, end_limit + 1):
        row = d.iloc[i]
        if pd.isna(row["ATR"]):
            continue

        long_sig = bool(row.get("long_sig", False))
        short_sig = bool(row.get("short_sig", False))

        pwin = None
        if use_ml and proba is not None:
            try:
                pwin = float(proba[i])
            except Exception:
                pwin = None

        def allowed():
            if not use_ml:
                return True
            if pwin is None or not np.isfinite(pwin):
                return False
            return pwin >= float(proba_thresh)

        if fill_next_open:
            fill_row = d.iloc[i + 1]
            fill_price = float(fill_row["open"])
            fill_time = fill_row["dt"]
        else:
            fill_price = float(row["close"])
            fill_time = row["dt"]

        if pos == 0:
            bars_in_trade = 0
            trail = None

            if long_sig and allowed():
                pos = 1
                entry = float(fill_price)
                entry_time = fill_time
                signal_time = row["dt"]
                signal_proba = pwin
                signal_tag = str(row.get("sig_tag", ""))

                sl = entry - atr_mult * float(row["ATR"])
                tp = entry + rr * (entry - sl)

                if exit_model == "ATR Trailing Stop":
                    trail = entry - trail_atr_mult * float(row["ATR"])

            elif both_dirs and short_sig and allowed():
                pos = -1
                entry = float(fill_price)
                entry_time = fill_time
                signal_time = row["dt"]
                signal_proba = pwin
                signal_tag = str(row.get("sig_tag", ""))

                sl = entry + atr_mult * float(row["ATR"])
                tp = entry - rr * (sl - entry)

                if exit_model == "ATR Trailing Stop":
                    trail = entry + trail_atr_mult * float(row["ATR"])

        else:
            if fill_next_open and row["dt"] < entry_time:
                continue

            bars_in_trade += 1

            if exit_model == "ATR Trailing Stop" and trail is not None:
                if pos == 1:
                    trail = max(trail, float(row["close"]) - trail_atr_mult * float(row["ATR"]))
                    sl_eff = max(sl, trail)
                else:
                    trail = min(trail, float(row["close"]) + trail_atr_mult * float(row["ATR"]))
                    sl_eff = min(sl, trail)
            else:
                sl_eff = sl

            if pos == 1:
                stop_hit = row["low"] <= sl_eff
                tp_hit = row["high"] >= tp
                opp_hit = exit_on_opp and short_sig
            else:
                stop_hit = row["high"] >= sl_eff
                tp_hit = row["low"] <= tp
                opp_hit = exit_on_opp and long_sig

            ema_flip_hit = False
            time_hit = False

            if exit_model == "EMA Flip Exit":
                if "EMA20" in d.columns and "EMA50" in d.columns:
                    if pos == 1:
                        ema_flip_hit = float(row["EMA20"]) < float(row["EMA50"])
                    else:
                        ema_flip_hit = float(row["EMA20"]) > float(row["EMA50"])

            if exit_model == "Time Stop":
                time_hit = bars_in_trade >= int(time_stop_bars)

            if exit_model == "Opposite Signal":
                if pos == 1:
                    opp_hit = short_sig
                else:
                    opp_hit = long_sig

            should_exit = stop_hit or tp_hit or opp_hit or ema_flip_hit or time_hit

            if should_exit:
                if stop_hit and tp_hit:
                    if intrabar_is_best:
                        stop_hit = False
                    else:
                        tp_hit = False

                if stop_hit:
                    exit_price = float(sl_eff)
                    pnl = -float(risk)
                    outcome = "SL" if exit_model != "ATR Trailing Stop" else "TRAIL_SL"
                elif tp_hit:
                    exit_price = float(tp)
                    pnl = float(risk) * float(rr)
                    outcome = "TP"
                elif ema_flip_hit:
                    exit_price = float(row["close"])
                    if pos == 1:
                        rdist = entry - sl
                        pnl = float(risk) * ((exit_price - entry) / rdist) if rdist > 0 else 0.0
                    else:
                        rdist = sl - entry
                        pnl = float(risk) * ((entry - exit_price) / rdist) if rdist > 0 else 0.0
                    outcome = "EMA_FLIP"
                elif time_hit:
                    exit_price = float(row["close"])
                    if pos == 1:
                        rdist = entry - sl
                        pnl = float(risk) * ((exit_price - entry) / rdist) if rdist > 0 else 0.0
                    else:
                        rdist = sl - entry
                        pnl = float(risk) * ((entry - exit_price) / rdist) if rdist > 0 else 0.0
                    outcome = "TIME"
                else:
                    exit_price = float(row["close"])
                    if pos == 1:
                        rdist = entry - sl
                        pnl = float(risk) * ((exit_price - entry) / rdist) if rdist > 0 else 0.0
                    else:
                        rdist = sl - entry
                        pnl = float(risk) * ((entry - exit_price) / rdist) if rdist > 0 else 0.0
                    outcome = "Opp"

                if pos == 1:
                    rdist = entry - sl
                    r_mult = (exit_price - entry) / rdist if rdist > 0 else 0.0
                else:
                    rdist = sl - entry
                    r_mult = (entry - exit_price) / rdist if rdist > 0 else 0.0

                trades.append(
                    {
                        "side": "LONG" if pos == 1 else "SHORT",
                        "sig_tag": signal_tag,
                        "signal_time": signal_time,
                        "entry_time": entry_time,
                        "exit_time": row["dt"],
                        "entry_price": float(entry),
                        "exit_price": float(exit_price),
                        "sl": float(sl),
                        "tp": float(tp),
                        "rr": float(rr),
                        "risk": float(risk),
                        "p_win": float(signal_proba) if signal_proba is not None and np.isfinite(signal_proba) else np.nan,
                        "r_mult": float(r_mult),
                        "pnl": float(pnl),
                        "outcome": outcome,
                        "exit_model": exit_model,
                        "bars_in_trade": int(bars_in_trade),
                        "score": float(row.get("score", np.nan)),
                    }
                )

                pos = 0
                entry = sl = tp = None
                entry_time = None
                signal_time = None
                signal_proba = None
                signal_tag = None
                bars_in_trade = 0
                trail = None

    return d, pd.DataFrame(trades)


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    df_view,
    atr_len,
    atr_mult,
    rr,
    risk,
    both_dirs,
    exit_on_opp,
    fill_next_open,
    intrabar_rule,
    exit_model,
    trail_atr_mult,
    time_stop_bars,
    use_ml,
    proba,
    proba_thresh,
):
    return backtest(
        df_view,
        atr_len,
        atr_mult,
        rr,
        risk,
        both_dirs,
        exit_on_opp,
        fill_next_open=fill_next_open,
        intrabar_rule=intrabar_rule,
        exit_model=exit_model,
        trail_atr_mult=trail_atr_mult,
        time_stop_bars=time_stop_bars,
        use_ml=use_ml,
        proba=proba,
        proba_thresh=proba_thresh,
    )


# =====================
# WALK-FORWARD ML
# =====================
def walk_forward_proba(
    d_sig: pd.DataFrame,
    X_all: np.ndarray,
    train_bars: int,
    test_bars: int,
    atr_len: int,
    atr_mult: float,
    rr: float,
    risk: float,
    both_dirs: bool,
    exit_on_opp: bool,
    fill_next_open: bool,
    intrabar_rule: str,
    exit_model: str,
    trail_atr_mult: float,
    time_stop_bars: int,
):
    n = len(d_sig)
    proba = np.full(n, np.nan, dtype=float)
    trained_windows = 0
    used_trades = 0

    start = max(train_bars, 200)
    t = start
    while t < n - 50:
        train_start = max(0, t - train_bars)
        train_end = t
        test_end = min(n, t + test_bars)

        d_train = d_sig.iloc[train_start:train_end].copy()

        _, tt = backtest(
            d_train,
            atr_len=atr_len,
            atr_mult=atr_mult,
            rr=rr,
            risk=risk,
            both_dirs=both_dirs,
            exit_on_opp=exit_on_opp,
            fill_next_open=fill_next_open,
            intrabar_rule=intrabar_rule,
            exit_model=exit_model,
            trail_atr_mult=trail_atr_mult,
            time_stop_bars=time_stop_bars,
            use_ml=False,
        )
        tt = tt[tt["outcome"].isin(["TP", "SL", "TRAIL_SL"])].copy()
        if len(tt) < 120:
            t = test_end
            continue

        dt_list = d_train["dt"].tolist()
        dt_to_local = {dt: i for i, dt in enumerate(dt_list)}
        idxs = []
        for stime in tt["signal_time"].tolist():
            idx = dt_to_local.get(stime, None)
            if idx is None:
                idx = int(np.searchsorted(np.array(dt_list, dtype="datetime64[ns]"), np.datetime64(stime)))
                idx = max(0, min(idx, len(d_train) - 1))
            idxs.append(idx)

        X_train = X_all[train_start:train_end][np.array(idxs, dtype=int)]
        y = (tt["r_mult"].to_numpy(dtype=float) > 0).astype(float)

        model = fit_logreg(X_train, y, lr=0.2, steps=650, l2=1e-2, seed=7)
        if model is None:
            t = test_end
            continue

        X_test = X_all[t:test_end]
        p = predict_logreg_proba(X_test, model)
        proba[t:test_end] = p

        trained_windows += 1
        used_trades += len(tt)
        t = test_end

    return proba, trained_windows, used_trades


# =====================
# SESSION OVERLAYS
# =====================
def add_session_overlays(fig, dts: pd.Series, show_asia=True, show_london=True, show_ny=True):
    """
    Adds simple UTC session boxes per day.
    Times (UTC):
      - Asia:   00:00-06:00
      - London: 07:00-10:00
      - NY:     13:00-16:00
    """
    if dts.empty:
        return fig

    days = pd.to_datetime(dts).dt.floor("D").unique()
    sessions = []
    if show_asia:
        sessions.append(("Asia", pd.Timedelta(hours=0), pd.Timedelta(hours=6)))
    if show_london:
        sessions.append(("London", pd.Timedelta(hours=7), pd.Timedelta(hours=10)))
    if show_ny:
        sessions.append(("NY", pd.Timedelta(hours=13), pd.Timedelta(hours=16)))

    for day in days:
        day = pd.Timestamp(day)
        for name, start_off, end_off in sessions:
            x0 = day + start_off
            x1 = day + end_off
            fig.add_vrect(
                x0=x0,
                x1=x1,
                opacity=0.10,
                layer="below",
                line_width=0,
                annotation_text=name,
                annotation_position="top left",
            )
    return fig


# =====================
# BEGINNER HELP
# =====================
def explain(label: str, what: str, up: str, down: str):
    st.caption(f"**{label}** — {what}")
    st.caption(f"⬆️ Increase it: {up}")
    st.caption(f"⬇️ Decrease it: {down}")


# =====================
# UI
# =====================
st.title("BTC Strategy Trade Visualiser (v4.9) — beginner help + strategies + scoring + walk-forward ML")

with st.sidebar:
    st.header("Upload Data")
    files = st.file_uploader("Binance BTCUSDT ZIP / CSV", type=["zip", "csv"], accept_multiple_files=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.rerun()
    with colB:
        st.write("")

    st.divider()
    beginner_mode = st.toggle("Beginner mode (show explanations)", True, help="Turn ON to see plain-English help and what each slider changes.")

    st.divider()
    st.header("Help / Glossary")
    with st.expander("What do these settings mean? (click to expand)", expanded=False):
        st.markdown(
            """
**Core indicators**
- **SMA (Simple Moving Average):** Average of the last N closes. Smooths noise.
- **EMA (Exponential Moving Average):** Like SMA but reacts faster to new prices.
- **RSI (Relative Strength Index):** 0–100 momentum gauge. Often >70 “overbought”, <30 “oversold”.
- **ROC (Rate of Change):** % change over N candles. Measures speed of price movement.
- **ATR (Average True Range):** Typical candle movement (volatility). Higher ATR = more volatile.
- **VWAP (Volume Weighted Avg Price):** “Fair price” weighted by volume. Often used for mean reversion.
- **Bollinger Bands:** Moving average ± K standard deviations. Band width shows volatility.

**Trade logic**
- **RR (Risk:Reward):** If RR=2.0 you aim to make 2x what you risk.
- **Risk per trade ($):** Fixed $ loss if stop-loss is hit.
- **Enter next candle open:** More realistic (signal confirmed at candle close, fill at next open).
- **TP/SL hit same candle:** Candle charts don’t show the exact order of price inside the candle.

**Confluence scoring**
- **Score:** Weighted average (0–100) of conditions (trend, RSI, volume, etc).
- **Min score:** Only take trades where score ≥ this threshold.
- **Weights:** How important each condition is for the score.

**Sessions (UTC overlays)**
- **Asia / London / NY boxes:** Helps see if a strategy works better in certain time windows.
"""
        )

    st.divider()
    st.header("Timeframe")
    tf_label_to_rule = {"1m": "1T", "5m": "5T", "10m": "10T", "15m": "15T", "1h": "1H", "4h": "4H"}
    timeframe = st.selectbox("Chart/Backtest timeframe", list(tf_label_to_rule.keys()), index=3,
                             help="This changes BOTH the chart and the backtest timeframe.")

    st.divider()
    st.header("Strategy Engine")
    strategy_name = st.selectbox(
        "Choose strategy",
        STRATEGIES,
        index=0,
        help="Pick which strategy generates entry signals. Exits are configured below."
    )

    if beginner_mode:
        STRAT_EXPLAIN = {
            "MA Crossover": "Buys when fast SMA crosses above slow SMA, sells/shorts on opposite. Best in trends; sideways chop hurts it.",
            "Momentum (ROC+VOL+RSI)": "Trades when price is moving fast (ROC), volume expands, and RSI confirms momentum.",
            "Trend Following (EMA pullback)": "Trades in trend direction after price pulls back to EMA area; exits on trend flip.",
            "Range Breakout": "Finds a tight range and trades the breakout beyond the range boundary.",
            "Bollinger Squeeze Breakout": "Trades volatility expansion after a ‘squeeze’ (very tight Bollinger Bands).",
            "RSI Mean Reversion": "Buys oversold and sells overbought aiming for a snap-back to normal.",
            "VWAP Reversion": "Trades when price is far from VWAP expecting it to revert toward ‘fair price’.",
            "Liquidity Sweep (basic)": "Looks for stop-hunts above/below recent equal highs/lows and trades the reversal.",
        }
        st.info(f"**How this strategy works:** {STRAT_EXPLAIN.get(strategy_name,'')}")

    # Strategy hints (quick defaults)
    DEFAULTS_HINT = {
        "MA Crossover": "Typical: Fast=10, Slow=20, Trend=200. Works better on higher TFs; chop hurts.",
        "Momentum (ROC+VOL+RSI)": "Typical: ROC len 20, ROC thr 0.25–0.60, RSI 14, Vol MA 20.",
        "Trend Following (EMA pullback)": "EMA20/EMA50 used internally. RSI filter mild (55/45).",
        "Range Breakout": "Range lookback 30–80. Require close outside for cleaner signals.",
        "Bollinger Squeeze Breakout": "BB len 20, k=2.0, squeeze width 1–3%.",
        "RSI Mean Reversion": "RSI 14. Time stop helps avoid trades that never revert.",
        "VWAP Reversion": "Daily VWAP on. Dev 0.7–2.0%.",
        "Liquidity Sweep (basic)": "Best near equal highs/lows. Combine with higher TF trend bias.",
    }
    st.caption(f"**Strategy note:** {DEFAULTS_HINT.get(strategy_name,'')}")

    st.subheader("Core Params (some strategies)")
    fast = st.number_input("Fast SMA", 2, 500, 10, help="Short-term average price. Controls how quickly signals react.")
    if beginner_mode:
        explain(
            "Fast SMA",
            "A smooth line of the last N closes. It represents short-term direction.",
            "Signals react slower; fewer trades; usually cleaner but later entries.",
            "Signals react faster; more trades; earlier entries but more false signals."
        )

    slow = st.number_input("Slow SMA", 3, 1000, 20, help="Longer-term average price. Used as the trend reference for crossover.")
    if beginner_mode:
        explain(
            "Slow SMA",
            "A slower moving average. When fast crosses it, that’s the main crossover signal.",
            "Fewer signals; stronger trend confirmation; later entries.",
            "More signals; faster flips; can get chopped in sideways markets."
        )

    trend_len = st.number_input("Trend SMA", 10, 2000, 200, help="Big trend filter baseline. Used by some strategies to avoid countertrend trades.")
    if beginner_mode:
        explain(
            "Trend SMA",
            "A big 'market bias' line. Often: take longs above it and shorts below it.",
            "More long-term bias; filters out more countertrend trades; fewer trades.",
            "Less strict bias; more trades; more chance of trading against the bigger move."
        )

    st.subheader("Strategy Params")
    rsi_len = st.number_input("RSI Length", 2, 100, 14, help="RSI lookback candles. 14 is common.")
    if beginner_mode:
        explain(
            "RSI Length",
            "RSI measures momentum (how strong recent moves are).",
            "RSI is smoother; fewer signals; less noise.",
            "RSI is twitchier; more signals; more fakeouts."
        )

    roc_len = st.number_input("ROC Length", 2, 200, 20, help="ROC lookback candles. Measures % change over N candles.")
    if beginner_mode:
        explain(
            "ROC Length",
            "ROC measures speed: % change over the last N candles.",
            "ROC becomes slower; only sustained moves trigger (fewer momentum trades).",
            "ROC becomes faster; catches bursts earlier but triggers more often."
        )

    momentum_roc_thresh = st.number_input(
        "Momentum ROC threshold (%)", 0.05, 5.0, 0.25, 0.05,
        help="Minimum ROC speed needed to classify as momentum."
    )
    if beginner_mode:
        explain(
            "Momentum ROC threshold",
            "How strong the price move must be to count as momentum.",
            "Fewer trades; when it triggers the move is usually stronger.",
            "More trades; includes weaker moves; more false momentum."
        )

    vol_len = st.number_input("Volume MA Length", 2, 500, 20, help="Volume average length. Used to detect volume expansion.")
    if beginner_mode:
        explain(
            "Volume MA Length",
            "Baseline volume level. 'Volume expansion' means volume is high vs this average.",
            "More stable volume baseline; fewer 'volume spike' detections.",
            "More sensitive; more spikes detected; can be noisier on low-volume periods."
        )

    range_len = st.number_input("Range lookback", 5, 500, 50, help="How many candles define the range for breakouts.")
    if beginner_mode:
        explain(
            "Range lookback",
            "Defines the box (range). Breakout happens when price exits it.",
            "Bigger range = fewer breakouts; stronger levels; slower to update.",
            "Smaller range = more breakouts; weaker levels; more noise."
        )

    break_close = st.toggle("Range breakout requires CLOSE outside", True, help="ON = safer but later. OFF = earlier but noisier.")
    if beginner_mode:
        explain(
            "Close outside (breakout)",
            "Whether breakout must be confirmed by candle close beyond the range.",
            "Cleaner signals; fewer fakeouts; entries are later.",
            "Earlier entries; more signals; higher fakeout risk."
        )

    bb_len = st.number_input("BB Length", 5, 300, 20, help="Bollinger band length (usually 20).")
    if beginner_mode:
        explain(
            "BB Length",
            "How far back Bollinger Bands look to estimate normal price movement.",
            "Smoother bands; fewer squeezes; slower reactions.",
            "More reactive bands; more squeezes; more signals but noisier."
        )

    bb_k = st.number_input("BB StdDev", 1.0, 4.0, 2.0, 0.25, help="Band width multiplier. 2.0 is common.")
    if beginner_mode:
        explain(
            "BB StdDev (K)",
            "How wide bands are around the average.",
            "Wider bands: fewer breakouts; signals require stronger moves.",
            "Narrower bands: more breakouts; more frequent but weaker signals."
        )

    squeeze_pct = st.number_input(
        "Squeeze threshold (BB width %)", 0.1, 20.0, 2.0, 0.1,
        help="When BB width % is below this, market is 'compressed'."
    )
    if beginner_mode:
        explain(
            "Squeeze threshold",
            "Defines what counts as 'low volatility'. Lower means stricter.",
            "Stricter squeeze: fewer setups, often better quality.",
            "Looser squeeze: more setups, but includes less meaningful compression."
        )

    use_session_vwap = st.toggle("VWAP: session reset (daily)", True, help="ON = VWAP resets each day. OFF = anchored across whole range.")
    if beginner_mode:
        explain(
            "VWAP daily reset",
            "Whether VWAP restarts each day (common) or stays anchored from the start of your dataset.",
            "Daily VWAP: clearer intraday mean; better for day-style reversion logic.",
            "Anchored VWAP: tracks long-period 'fair value'; slower mean reversion signals."
        )

    vwap_dev = st.number_input("VWAP deviation trigger (%)", 0.1, 10.0, 1.0, 0.1, help="How far from VWAP before you consider it 'stretched'.")
    if beginner_mode:
        explain(
            "VWAP deviation trigger",
            "How far price must be from VWAP before you take a mean reversion signal.",
            "Fewer trades; only very stretched moves; often higher win rate but fewer opportunities.",
            "More trades; smaller deviations; can get run over in strong trends."
        )

    st.divider()
    st.header("Confluence Scoring")
    st.caption("Signals only taken if score ≥ min score. Score = weighted average of components.")

    min_score = st.slider("Min score to take trade", 0.0, 100.0, 55.0, 1.0, help="Only take signals if score is high enough.")
    if beginner_mode:
        explain(
            "Min score",
            "This is your 'quality bar'. A trade only happens if conditions line up strongly enough.",
            "Fewer trades; higher average quality; may miss some moves.",
            "More trades; lower average quality; more noise/loss streaks."
        )

    # weights
    w_trend = st.slider("Weight: Trend", 0.0, 5.0, 2.0, 0.5, help="How much trend alignment matters in the score.")
    if beginner_mode:
        explain(
            "Weight: Trend",
            "How much the score cares about trading with the bigger direction.",
            "Score becomes more trend-focused (filters countertrend trades).",
            "Trend matters less (more countertrend/mixed trades)."
        )

    w_rsi = st.slider("Weight: RSI", 0.0, 5.0, 1.5, 0.5, help="How much RSI contributes to the score.")
    if beginner_mode:
        explain(
            "Weight: RSI",
            "How much momentum / overbought-oversold condition impacts trade quality.",
            "Signals become more RSI-driven; fewer trades if RSI isn't confirming.",
            "RSI matters less; strategy relies more on other conditions."
        )

    w_vol = st.slider("Weight: Volume", 0.0, 5.0, 1.5, 0.5, help="How much volume expansion/exhaustion matters.")
    if beginner_mode:
        explain(
            "Weight: Volume",
            "How much you require volume to 'support' the move.",
            "Fewer trades; stronger moves preferred; filters low-volume noise.",
            "More trades; volume confirmation is less important."
        )

    w_strength = st.slider("Weight: Strength/Distance", 0.0, 5.0, 1.5, 0.5, help="Break strength / MA separation / distance-from-level.")
    if beginner_mode:
        explain(
            "Weight: Strength/Distance",
            "How much you reward 'decisive movement' away from key levels.",
            "Trades require stronger breaks / more separation; fewer but often cleaner.",
            "Allows weaker breaks; more trades; more fakeouts."
        )

    w_compression = st.slider("Weight: Compression", 0.0, 5.0, 1.0, 0.5, help="How much low-volatility compression matters.")
    if beginner_mode:
        explain(
            "Weight: Compression",
            "Rewards setups after quiet periods (ranges/squeezes).",
            "Favors squeeze/range setups; fewer trades but often better expansions.",
            "Compression matters less; strategy can trigger in messy conditions."
        )

    w_pullback = st.slider("Weight: Pullback", 0.0, 5.0, 1.0, 0.5, help="How much pullback quality matters (trend strategies).")
    if beginner_mode:
        explain(
            "Weight: Pullback",
            "Rewards price returning to a 'value area' before continuing the trend.",
            "More patient entries; fewer trades; can improve win rate.",
            "More aggressive entries; more trades; can enter too early."
        )

    w_roc = st.slider("Weight: ROC", 0.0, 5.0, 2.0, 0.5, help="How much speed (ROC) matters in momentum strategies.")
    if beginner_mode:
        explain(
            "Weight: ROC",
            "How much you care about 'speed' of price movement.",
            "More momentum-only trades; fewer but stronger moves.",
            "Less emphasis on speed; more trades even when movement is mild."
        )

    w_wick = st.slider("Weight: Wick/Rejection", 0.0, 5.0, 1.0, 0.5, help="How much wick rejection matters (liquidity sweeps).")
    if beginner_mode:
        explain(
            "Weight: Wick/Rejection",
            "Rewards long wicks / rejections that suggest a stop-hunt reversal.",
            "Fewer sweep trades; only strong rejection candles count.",
            "More sweep trades; weaker rejection allowed."
        )

    w_dev = st.slider("Weight: Deviation", 0.0, 5.0, 2.0, 0.5, help="How much distance-from-mean matters (VWAP/mean reversion).")
    if beginner_mode:
        explain(
            "Weight: Deviation",
            "Rewards price being far from the mean (VWAP/bands) before taking reversion trades.",
            "Waits for bigger stretch; fewer trades; often safer reversion attempts.",
            "More trades on small stretches; higher risk of trend continuation."
        )

    weights = {
        "trend": w_trend,
        "rsi": w_rsi,
        "vol": w_vol,
        "strength": w_strength,
        "compression": w_compression,
        "pullback": w_pullback,
        "roc": w_roc,
        "wick": w_wick,
        "dev": w_dev,
        "ma_sep": w_strength,
        "reclaim": w_strength,
    }

    st.divider()
    st.header("Risk / Execution")
    atr_len = st.number_input("ATR Length", 2, 100, 14, help="ATR lookback candles. Measures typical movement (volatility).")
    if beginner_mode:
        explain(
            "ATR Length",
            "How many candles ATR uses to estimate typical candle movement.",
            "Smoother ATR; stops adapt slower; less jumpy.",
            "Faster ATR response; stops adapt quicker to changing volatility."
        )

    atr_mult = st.number_input("ATR Multiplier", 0.5, 10.0, 2.5, help="Stop distance = ATR × multiplier.")
    if beginner_mode:
        explain(
            "ATR Multiplier",
            "Sets how wide the stop is based on volatility.",
            "Wider stops: fewer stop-outs; but you tolerate bigger move against you.",
            "Tighter stops: better R:R potential; more stop-outs."
        )

    rr = st.number_input("Risk Reward (RR)", 0.5, 10.0, 2.0, help="TP distance compared to stop distance (RR=2 aims for 2R).")
    if beginner_mode:
        explain(
            "Risk Reward (RR)",
            "How far the take-profit is relative to the stop distance.",
            "Bigger winners but fewer reach TP (win rate often drops).",
            "More TPs hit but smaller wins (needs higher win rate)."
        )

    risk = st.number_input("Risk per trade ($)", 1.0, 1_000_000.0, 100.0, help="Fixed $ loss if SL is hit (backtest uses this).")
    if beginner_mode:
        explain(
            "Risk per trade ($)",
            "Backtest assumes you lose this amount when SL hits (like risking $100 each trade).",
            "PnL swings bigger (more volatility).",
            "PnL swings smaller (more stable)."
        )

    both_dirs = st.toggle("Trade Long & Short", True, help="ON = allows both longs and shorts. OFF = longs only.")
    exit_on_opp = st.toggle("Exit on opposite signal (extra)", False, help="If ON, also exits when opposite signal happens (in addition to exit model).")

    fill_next_open = st.toggle("Enter on next candle OPEN (more realistic)", True, help="Signal is confirmed at close; fill happens next candle open.")
    if beginner_mode:
        explain(
            "Enter next candle open",
            "Prevents 'perfect fills' inside the signal candle. More realistic execution.",
            "More realistic; often slightly worse entries; less inflated results.",
            "More optimistic; can look better than real trading."
        )

    intrabar_rule = st.selectbox(
        "If TP & SL hit same candle",
        ["Worst case (SL first)", "Best case (TP first)"],
        index=0,
        help="Candles don’t show the internal path. Choose conservative or optimistic assumption."
    )
    if beginner_mode:
        explain(
            "TP/SL same candle",
            "If both TP and SL are inside one candle, you don't know which happened first.",
            "Worst-case is conservative realism (stress test).",
            "Best-case is optimistic (use carefully)."
        )

    st.divider()
    st.header("Exit Model")
    exit_override = st.selectbox("Exit model", EXIT_MODELS, index=0, help="AUTO chooses a sensible exit per strategy; override to compare exits.")
    trail_atr_mult = st.number_input("Trailing ATR multiple", 0.5, 10.0, 2.0, 0.25, help="Used by ATR Trailing Stop exit.")
    time_stop_bars = st.number_input("Time stop (bars)", 5, 5000, 48, help="Used by Time Stop exit (closes after N candles).")

    if beginner_mode:
        st.caption("**Exit model quick guide:**")
        st.caption("- **TP/SL only:** exit only when TP or SL hits.")
        st.caption("- **ATR Trailing Stop:** moves stop behind price as it moves in your favor.")
        st.caption("- **EMA Flip Exit:** exits when EMA20/EMA50 trend flips (trend-following style).")
        st.caption("- **Time Stop:** exits after N candles (helps mean reversion strategies).")
        st.caption("- **Opposite Signal:** exits when the strategy gives the opposite signal.")

    st.divider()
    st.header("ML Probability Filter")
    use_ml = st.toggle("Enable ML filter (score each signal)", False, help="Adds a P(win) model and filters out low-probability signals.")
    ml_mode = st.selectbox("ML mode", ["Single-split (fast)", "Walk-forward (realistic)"], index=1,
                           help="Walk-forward trains on past windows and predicts forward windows (no leakage).")
    train_frac = st.slider("Single-split training fraction", 0.3, 0.9, 0.7, help="Only used in Single-split mode.")
    wf_train_bars = st.number_input("Walk-forward train bars", 200, 20000, 3000, help="How many candles ML learns from each training window.")
    wf_test_bars = st.number_input("Walk-forward test bars", 50, 5000, 500, help="How many candles ML predicts forward each step.")
    proba_thresh = st.slider("Only take trades if P(win) ≥", 0.50, 0.80, 0.55, 0.01, help="Minimum ML probability required to accept a signal.")

    if beginner_mode:
        explain(
            "ML filter (P(win))",
            "ML tries to learn patterns that tend to win vs lose. It DOES NOT predict price. It filters signals.",
            "Higher threshold: fewer trades but higher average quality.",
            "Lower threshold: more trades, but more low-quality signals pass."
        )

    st.divider()
    st.header("Chart")
    show_indicators = st.toggle("Show strategy indicator lines", True, help="Shows lines like SMA/EMA/BB/VWAP (when available).")
    show_selected_sl_tp = st.toggle("Show SL/TP for selected trade", True)
    show_ml_overlay = st.toggle("Show ML P(win) markers on signals", True)

    st.subheader("Sessions (UTC overlays)")
    show_sessions = st.toggle("Show session overlays", True, help="Draw Asia/London/NY boxes on the chart for each day.")
    show_asia = st.toggle("Asia (00–06)", True)
    show_london = st.toggle("London (07–10)", True)
    show_ny = st.toggle("NY (13–16)", True)

    st.divider()
    st.header("Performance")
    run_now = st.button("Run backtest")


# =====================
# MAIN FLOW
# =====================
if not files:
    st.info("Upload Binance BTCUSDT files to begin.")
    st.stop()

file_payload = [(f.name, f.getvalue()) for f in files]
df = load_files_cached(file_payload)

if df.empty:
    st.error("No valid Binance data loaded.")
    st.stop()

base_min = infer_base_minutes(df)
st.caption(
    f"Loaded rows: {len(df):,} | Data range: {df['dt'].min()} → {df['dt'].max()} | Base ~ {base_min:.2f} min"
    if base_min
    else ""
)

min_dt = df["dt"].min().to_pydatetime()
max_dt = df["dt"].max().to_pydatetime()

date_range = st.slider("Date range (UTC)", min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt))
df_view_full = df[(df["dt"] >= date_range[0]) & (df["dt"] <= date_range[1])].copy()

target_rule = tf_label_to_rule[timeframe]
target_minutes = {"1T": 1, "5T": 5, "10T": 10, "15T": 15, "1H": 60, "4H": 240}[target_rule]

if base_min is None:
    st.error("Could not infer the base timeframe from your data.")
    st.stop()

if target_minutes < (base_min - 0.01):
    st.error(
        f"Your uploaded data is ~{base_min:.0f}m candles. "
        f"You cannot create {timeframe} from that. "
        f"Download and upload {timeframe} (or smaller, e.g. 1m) data."
    )
    st.stop()

df_tf = df_view_full.copy() if abs(base_min - target_minutes) < 0.5 else resample_cached(df_view_full, target_rule)
if df_tf.empty:
    st.error("No candles after timeframe conversion in this date range.")
    st.stop()

if "has_run" not in st.session_state:
    st.session_state.has_run = False
if not run_now and st.session_state.has_run is False:
    run_now = True
if not run_now:
    st.info("Adjust settings, then click **Run backtest**.")
    st.stop()
st.session_state.has_run = True

exit_model = exit_override
if exit_model == "AUTO":
    exit_model = DEFAULT_EXIT_FOR_STRAT.get(strategy_name, "TP/SL only")

# signals + score
d_sig = compute_signals(
    df_tf,
    strategy_name=strategy_name,
    fast=int(fast),
    slow=int(slow),
    trend_len=int(trend_len),
    rsi_len=int(rsi_len),
    roc_len=int(roc_len),
    vol_len=int(vol_len),
    range_len=int(range_len),

# Sanity-check signals dataframe before running ML/backtest
if d_sig is None or (not isinstance(d_sig, pd.DataFrame)) or d_sig.empty:
    st.error(
        "Signals dataframe is empty/invalid. This can happen if the timeframe/date range produces no candles, "
        "or if you deployed an older file missing `return x` in compute_signals. "
        "Try: (1) widen the date range, (2) switch timeframe to match your data, (3) click Clear cache, then reload."
    )
    st.stop()

    break_close=bool(break_close),
    bb_len=int(bb_len),
    bb_k=float(bb_k),
    squeeze_pct=float(squeeze_pct),
    use_session_vwap=bool(use_session_vwap),
    vwap_dev=float(vwap_dev),
    momentum_roc_thresh=float(momentum_roc_thresh),
    weights=weights,
    min_score=float(min_score),
)

# ML
proba = None
ml_info = None

if use_ml:
    X_all, feat_cols = build_ml_features(d_sig)

    if ml_mode == "Walk-forward (realistic)":
        proba, nwin, ntr = walk_forward_proba(
            d_sig,
            X_all,
            train_bars=int(wf_train_bars),
            test_bars=int(wf_test_bars),
            atr_len=int(atr_len),
            atr_mult=float(atr_mult),
            rr=float(rr),
            risk=float(risk),
            both_dirs=bool(both_dirs),
            exit_on_opp=bool(exit_on_opp),
            fill_next_open=bool(fill_next_open),
            intrabar_rule=str(intrabar_rule),
            exit_model=str(exit_model),
            trail_atr_mult=float(trail_atr_mult),
            time_stop_bars=int(time_stop_bars),
        )
        ml_info = f"Walk-forward ML: windows trained={nwin}, labeled trades used={ntr}. Features: {', '.join(feat_cols)}"
    else:
        cut_idx = int(len(d_sig) * float(train_frac))
        cut_idx = max(50, min(cut_idx, len(d_sig) - 50))
        train_end_dt = d_sig["dt"].iloc[cut_idx]
        d_train = d_sig[d_sig["dt"] <= train_end_dt].copy()

        _, tt = backtest(
            d_train,
            atr_len=int(atr_len),
            atr_mult=float(atr_mult),
            rr=float(rr),
            risk=float(risk),
            both_dirs=bool(both_dirs),
            exit_on_opp=bool(exit_on_opp),
            fill_next_open=bool(fill_next_open),
            intrabar_rule=str(intrabar_rule),
            exit_model=str(exit_model),
            trail_atr_mult=float(trail_atr_mult),
            time_stop_bars=int(time_stop_bars),
            use_ml=False,
        )
        tt = tt[tt["outcome"].isin(["TP", "SL", "TRAIL_SL"])].copy()

        if len(tt) < 150:
            st.warning("ML: Not enough TP/SL-labeled trades in training segment. Running without ML filter.")
            use_ml = False
            proba = None
        else:
            dt_list = d_train["dt"].tolist()
            dt_to_local = {dt: i for i, dt in enumerate(dt_list)}
            idxs = []
            for stime in tt["signal_time"].tolist():
                idx = dt_to_local.get(stime, None)
                if idx is None:
                    idx = int(np.searchsorted(np.array(dt_list, dtype="datetime64[ns]"), np.datetime64(stime)))
                    idx = max(0, min(idx, len(d_train) - 1))
                idxs.append(idx)

            X_train = X_all[: len(d_train)][np.array(idxs, dtype=int)]
            y = (tt["r_mult"].to_numpy(dtype=float) > 0).astype(float)

            model = fit_logreg(X_train, y, lr=0.2, steps=650, l2=1e-2, seed=7)
            if model is None:
                st.warning("ML: Training failed (insufficient clean rows). Running without ML filter.")
                use_ml = False
                proba = None
            else:
                proba = predict_logreg_proba(X_all, model)
                ml_info = f"Single-split ML trained ≤ {pd.Timestamp(train_end_dt):%Y-%m-%d}. Features: {', '.join(feat_cols)}"

if ml_info:
    st.info(ml_info)

# backtest
d_full, trades = run_backtest_cached(
    d_sig,
    int(atr_len),
    float(atr_mult),
    float(rr),
    float(risk),
    bool(both_dirs),
    bool(exit_on_opp),
    bool(fill_next_open),
    str(intrabar_rule),
    str(exit_model),
    float(trail_atr_mult),
    int(time_stop_bars),
    bool(use_ml),
    proba,
    float(proba_thresh),
)

wins = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
losses = int((trades["pnl"] < 0).sum()) if not trades.empty else 0
net = float(trades["pnl"].sum()) if not trades.empty else 0.0
wr = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
avg_r = float(trades["r_mult"].mean()) if (not trades.empty and "r_mult" in trades.columns) else 0.0
avg_p = float(trades["p_win"].mean()) if (not trades.empty and "p_win" in trades.columns) else np.nan
avg_score = float(trades["score"].mean()) if (not trades.empty and "score" in trades.columns) else np.nan

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Trades", len(trades))
c2.metric("Win rate", f"{wr:.1f}%")
c3.metric("Wins", wins)
c4.metric("Losses", losses)
c5.metric("Net PnL ($)", f"{net:.0f}")
c6.metric("Avg R", f"{avg_r:.2f}")
c7.metric("Avg P(win)", f"{avg_p:.2f}" if np.isfinite(avg_p) else "—")
c8.metric("Avg Score", f"{avg_score:.1f}" if np.isfinite(avg_score) else "—")

# trade selector
selected_idx = None
t = None
if not trades.empty:
    t = trades.reset_index(drop=True)
    labels = [
        f"#{i+1} {r.side} | {r.entry_time:%Y-%m-%d %H:%M} → {r.exit_time:%Y-%m-%d %H:%M} | {r.outcome} | "
        f"Score {r.score:.0f} | P(win) {r.p_win:.2f} | R {r.r_mult:+.2f} | {r.sig_tag} | {r.exit_model}"
        for i, r in t.iterrows()
    ]
    selected = st.selectbox("Select a trade to highlight", ["(none)"] + labels, index=0)
    if selected != "(none)":
        selected_idx = labels.index(selected)

# chart
d_chart = d_full.copy()

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=d_chart["dt"],
        open=d_chart["open"],
        high=d_chart["high"],
        low=d_chart["low"],
        close=d_chart["close"],
        name=f"BTCUSDT ({timeframe})",
    )
)

if show_sessions:
    fig = add_session_overlays(fig, d_chart["dt"], show_asia=show_asia, show_london=show_london, show_ny=show_ny)

if show_indicators:
    for col_name, label in [
        ("FAST", f"SMA {fast}"),
        ("SLOW", f"SMA {slow}"),
        ("TREND", f"SMA {trend_len}"),
        ("EMA20", "EMA20"),
        ("EMA50", "EMA50"),
        ("BB_U", "BB Upper"),
        ("BB_M", "BB Mid"),
        ("BB_L", "BB Lower"),
        ("VWAP", "VWAP"),
        ("R_H", "Range High"),
        ("R_L", "Range Low"),
    ]:
        if col_name in d_chart.columns:
            fig.add_trace(go.Scatter(x=d_chart["dt"], y=d_chart[col_name], mode="lines", name=label))

if t is not None and not t.empty:
    fig.add_trace(
        go.Scatter(
            x=t["entry_time"],
            y=t["entry_price"],
            mode="markers",
            name="Entries",
            marker=dict(symbol="triangle-up", size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["exit_time"],
            y=t["exit_price"],
            mode="markers",
            name="Exits",
            marker=dict(symbol="x", size=10),
        )
    )

if selected_idx is not None and t is not None:
    r = t.iloc[selected_idx]
    fig.add_trace(
        go.Scatter(
            x=[r["entry_time"], r["exit_time"]],
            y=[r["entry_price"], r["exit_price"]],
            mode="lines+markers",
            name="Selected trade",
            line=dict(width=4),
            marker=dict(size=11),
        )
    )
    if show_selected_sl_tp:
        fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["sl"], r["sl"]], mode="lines", name="Selected SL"))
        fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["tp"], r["tp"]], mode="lines", name="Selected TP"))

if show_ml_overlay and use_ml and (proba is not None) and ("long_sig" in d_chart.columns):
    sig_mask = (d_chart["long_sig"] | d_chart["short_sig"])
    sig_points = d_chart[sig_mask].copy()
    if len(sig_points) > 0:
        idx = sig_points.index.to_numpy()
        p = np.array(proba)[idx]
        fig.add_trace(
            go.Scatter(
                x=sig_points["dt"],
                y=sig_points["close"],
                mode="markers",
                name="ML P(win) @ signals",
                marker=dict(
                    size=9,
                    color=p,
                    cmin=0.0,
                    cmax=1.0,
                    colorscale="Viridis",
                    colorbar=dict(title="P(win)"),
                    symbol="circle",
                    opacity=0.9,
                ),
                text=[f"P(win)={pp:.2f}" if np.isfinite(pp) else "P(win)=NA" for pp in p],
                hovertemplate="%{text}<br>%{x}<br>Close=%{y}<extra></extra>",
            )
        )

fig.update_layout(height=720, dragmode="pan", xaxis_rangeslider_visible=False, hovermode="x unified")
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False})

st.subheader("Trades Table")
if trades.empty:
    st.write("No trades in this range with current settings.")
else:
    tshow = trades.copy()
    for col in ["signal_time", "entry_time", "exit_time"]:
        tshow[col] = pd.to_datetime(tshow[col]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(tshow, use_container_width=True, hide_index=True)

    st.download_button(
        "Download trades CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{timeframe}_{strategy_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )
