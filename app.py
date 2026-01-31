# BTC Strategy Trade Visualiser — app.py
# - Multi-strategy entries + strategy-specific exit defaults
# - Confluence scoring (weights + min score) with explainable components
# - Backtest with realistic entry (close->next open), intrabar TP/SL rule, multiple exits
# - ML filter (walk-forward or single-split) using simple logistic regression (no sklearn)
# - Trade drilldown: reason snapshot JSON + MFE/MAE + component contributions
# - UX: dropdown sections (expanders), presets, save/load settings JSON
# - Chart: Candles / Heikin Ashi / OHLC / Line / Area + lag controls (last N, markers cap)

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="BTC Strategy Trade Visualiser", layout="wide")


# ============================================================
# DATA LOADER
# ============================================================
def _read_binance_like_csv(fileobj) -> pd.DataFrame:
    df = pd.read_csv(fileobj, header=None)
    if df.shape[1] < 6:
        return pd.DataFrame()
    df = df.iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    return df


def load_files(files) -> pd.DataFrame:
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


# ============================================================
# TIMEFRAME / RESAMPLE
# ============================================================
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


# ============================================================
# INDICATORS
# ============================================================
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


# ============================================================
# CONFLUENCE SCORING (vectorized)
# ============================================================
def normalize_score_vec(parts_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if parts_df is None or parts_df.empty:
        return pd.Series(np.zeros(0))
    cols = list(parts_df.columns)
    if not cols:
        return pd.Series(np.zeros(len(parts_df)))
    w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
    w = np.where(w < 0, 0.0, w)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return pd.Series(np.zeros(len(parts_df)))
    m = parts_df.to_numpy(dtype=float)
    m = np.clip(m, 0.0, 1.0)
    s = (m @ w) / wsum
    return pd.Series(s * 100.0, index=parts_df.index)


# ============================================================
# STRATEGY ENGINE
# ============================================================
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

    # shared
    x["RSI"] = rsi(x["close"], rsi_len)
    x["ROC"] = roc(x["close"], roc_len)
    x["VOL_MA"] = x["volume"].rolling(vol_len).mean()
    x["VOL_EXP"] = x["volume"] > (x["VOL_MA"] * 1.3)

    x["long_sig"] = False
    x["short_sig"] = False
    x["raw_long"] = False
    x["raw_short"] = False
    x["sig_tag"] = ""
    x["score"] = 0.0
    x["score_components"] = ""

    def apply_score(mask_long: pd.Series, mask_short: pd.Series, parts_df: pd.DataFrame, tag_long: str, tag_short: str):
        x["raw_long"] = mask_long.astype(bool)
        x["raw_short"] = mask_short.astype(bool)

        parts_keys = list(parts_df.columns)
        for k in parts_keys:
            x[f"score_{k}"] = parts_df[k].astype(float)

        x["score"] = normalize_score_vec(parts_df, weights).astype(float).to_numpy()
        ok = x["score"] >= float(min_score)

        x["long_sig"] = mask_long & ok
        x["short_sig"] = mask_short & ok
        x.loc[x["long_sig"], "sig_tag"] = tag_long
        x.loc[x["short_sig"], "sig_tag"] = tag_short

        x["score_components"] = ",".join(parts_keys)

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
                "trend": (x["close"] > x["TREND"]).astype(float),
                "rsi": ((x["RSI"] - 50.0) / 50.0).clip(0, 1),
                "ma_sep": ((x["FAST"] - x["SLOW"]) / x["close"]).abs().clip(0, 0.01) / 0.01,
            },
            index=x.index,
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
            },
            index=x.index,
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
            },
            index=x.index,
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
            },
            index=x.index,
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
            },
            index=x.index,
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
            },
            index=x.index,
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
            },
            index=x.index,
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
            },
            index=x.index,
        )
        apply_score(long_raw, short_raw, parts, "LIQ_SWEEP_LONG", "LIQ_SWEEP_SHORT")

    return x


# ============================================================
# ML (logistic regression, no sklearn)
# ============================================================
def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X, y, lr=0.2, steps=650, l2=1e-2, seed=7):
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


# ============================================================
# BACKTEST + SIZING + REASON SNAPSHOT
# ============================================================
def compute_trade_risk(
    risk_mode: str,
    fixed_risk: float,
    equity: float,
    risk_pct: float,
    leverage_cap: float,
    min_notional: float,
    max_notional: float,
    entry: float,
    sl: float,
) -> tuple[float, float]:
    """
    Returns (risk_used_dollars, qty)
    qty is based on risk (distance to SL) and capped by leverage/notional.
    """
    equity = float(equity)
    fixed_risk = float(fixed_risk)
    risk_pct = float(risk_pct)
    leverage_cap = float(leverage_cap)
    min_notional = float(min_notional)
    max_notional = float(max_notional)

    if risk_mode == "% Equity":
        risk_used = max(0.0, equity * (risk_pct / 100.0))
    else:
        risk_used = max(0.0, fixed_risk)

    dist = abs(float(entry) - float(sl))
    if dist <= 0 or not np.isfinite(dist):
        return 0.0, 0.0

    qty = risk_used / dist

    # apply leverage cap / notional caps
    notional = qty * entry
    max_notional_by_leverage = equity * leverage_cap if leverage_cap > 0 else max_notional
    hard_cap = min(max_notional, max_notional_by_leverage) if max_notional > 0 else max_notional_by_leverage
    if hard_cap > 0 and notional > hard_cap:
        qty = hard_cap / entry
        notional = qty * entry

    if min_notional > 0 and notional < min_notional:
        qty = min_notional / entry
        notional = qty * entry
        # re-check hard cap after bump
        if hard_cap > 0 and notional > hard_cap:
            qty = hard_cap / entry

    # recompute risk_used from final qty to keep reporting consistent
    risk_used = qty * dist
    return float(risk_used), float(qty)


def backtest(
    df: pd.DataFrame,
    strategy_name: str,
    weights: dict[str, float],
    min_score: float,
    atr_len: int,
    atr_mult: float,
    rr: float,
    risk_mode: str,
    risk: float,
    start_equity: float,
    risk_pct: float,
    leverage_cap: float,
    min_notional: float,
    max_notional: float,
    both_dirs: bool,
    exit_on_opp: bool,
    fill_next_open: bool = True,
    intrabar_rule: str = "Worst case (SL first)",
    exit_model: str = "TP/SL only",
    trail_atr_mult: float = 2.0,
    time_stop_bars: int = 48,
    use_ml: bool = False,
    proba=None,
    proba_thresh: float = 0.55,
):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), pd.DataFrame()

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

    # excursion
    mfe = 0.0
    mae = 0.0
    entry_atr = None
    entry_snapshot_json = None
    qty = 0.0
    risk_used = 0.0
    equity = float(start_equity)

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
            mfe = 0.0
            mae = 0.0
            entry_snapshot_json = None
            qty = 0.0
            risk_used = 0.0

            if long_sig and allowed():
                pos = 1
                entry = float(fill_price)
                entry_time = fill_time
                signal_time = row["dt"]
                signal_proba = pwin
                signal_tag = str(row.get("sig_tag", ""))

                sl = entry - atr_mult * float(row["ATR"])
                tp = entry + rr * (entry - sl)

                # sizing
                risk_used, qty = compute_trade_risk(
                    risk_mode=risk_mode,
                    fixed_risk=risk,
                    equity=equity,
                    risk_pct=risk_pct,
                    leverage_cap=leverage_cap,
                    min_notional=min_notional,
                    max_notional=max_notional,
                    entry=entry,
                    sl=sl,
                )

                if exit_model == "ATR Trailing Stop":
                    trail = entry - trail_atr_mult * float(row["ATR"])

                # snapshot
                entry_atr = float(row["ATR"])
                comp_names = str(row.get("score_components", "")).split(",") if "score_components" in row else []
                comp_names = [c for c in comp_names if c]
                components = {}
                contrib = {}
                for c in comp_names:
                    v = float(row.get(f"score_{c}", np.nan))
                    w = float(weights.get(c, 0.0))
                    components[c] = v
                    contrib[c] = (v * w)

                snapshot = {
                    "strategy": strategy_name,
                    "sig_tag": str(row.get("sig_tag", "")),
                    "signal_time": str(row["dt"]),
                    "entry_fill_mode": "next_open" if fill_next_open else "close",
                    "min_score": float(min_score),
                    "score": float(row.get("score", np.nan)),
                    "raw_long": bool(row.get("raw_long", False)),
                    "raw_short": bool(row.get("raw_short", False)),
                    "indicators": {
                        "RSI": float(row.get("RSI", np.nan)),
                        "ROC": float(row.get("ROC", np.nan)),
                        "VOL_EXP": bool(row.get("VOL_EXP", False)),
                        "ATR": float(row.get("ATR", np.nan)),
                    },
                    "components": components,
                    "weights": {k: float(v) for k, v in weights.items()} if isinstance(weights, dict) else {},
                    "contribution": contrib,
                    "sizing": {
                        "risk_mode": risk_mode,
                        "equity": float(equity),
                        "risk_used": float(risk_used),
                        "qty": float(qty),
                        "notional": float(qty * entry),
                        "leverage_cap": float(leverage_cap),
                        "min_notional": float(min_notional),
                        "max_notional": float(max_notional),
                    },
                }
                entry_snapshot_json = json.dumps(snapshot, ensure_ascii=False)

            elif both_dirs and short_sig and allowed():
                pos = -1
                entry = float(fill_price)
                entry_time = fill_time
                signal_time = row["dt"]
                signal_proba = pwin
                signal_tag = str(row.get("sig_tag", ""))

                sl = entry + atr_mult * float(row["ATR"])
                tp = entry - rr * (sl - entry)

                # sizing
                risk_used, qty = compute_trade_risk(
                    risk_mode=risk_mode,
                    fixed_risk=risk,
                    equity=equity,
                    risk_pct=risk_pct,
                    leverage_cap=leverage_cap,
                    min_notional=min_notional,
                    max_notional=max_notional,
                    entry=entry,
                    sl=sl,
                )

                if exit_model == "ATR Trailing Stop":
                    trail = entry + trail_atr_mult * float(row["ATR"])

                entry_atr = float(row["ATR"])
                comp_names = str(row.get("score_components", "")).split(",") if "score_components" in row else []
                comp_names = [c for c in comp_names if c]
                components = {}
                contrib = {}
                for c in comp_names:
                    v = float(row.get(f"score_{c}", np.nan))
                    w = float(weights.get(c, 0.0))
                    components[c] = v
                    contrib[c] = (v * w)

                snapshot = {
                    "strategy": strategy_name,
                    "sig_tag": str(row.get("sig_tag", "")),
                    "signal_time": str(row["dt"]),
                    "entry_fill_mode": "next_open" if fill_next_open else "close",
                    "min_score": float(min_score),
                    "score": float(row.get("score", np.nan)),
                    "raw_long": bool(row.get("raw_long", False)),
                    "raw_short": bool(row.get("raw_short", False)),
                    "indicators": {
                        "RSI": float(row.get("RSI", np.nan)),
                        "ROC": float(row.get("ROC", np.nan)),
                        "VOL_EXP": bool(row.get("VOL_EXP", False)),
                        "ATR": float(row.get("ATR", np.nan)),
                    },
                    "components": components,
                    "weights": {k: float(v) for k, v in weights.items()} if isinstance(weights, dict) else {},
                    "contribution": contrib,
                    "sizing": {
                        "risk_mode": risk_mode,
                        "equity": float(equity),
                        "risk_used": float(risk_used),
                        "qty": float(qty),
                        "notional": float(qty * entry),
                        "leverage_cap": float(leverage_cap),
                        "min_notional": float(min_notional),
                        "max_notional": float(max_notional),
                    },
                }
                entry_snapshot_json = json.dumps(snapshot, ensure_ascii=False)

        else:
            if fill_next_open and row["dt"] < entry_time:
                continue

            bars_in_trade += 1

            # excursion tracking
            if pos == 1:
                mfe = max(mfe, float(row["high"]) - float(entry))
                mae = max(mae, float(entry) - float(row["low"]))
            else:
                mfe = max(mfe, float(entry) - float(row["low"]))
                mae = max(mae, float(row["high"]) - float(entry))

            # trailing logic
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
                opp_hit = (short_sig if pos == 1 else long_sig)

            should_exit = stop_hit or tp_hit or opp_hit or ema_flip_hit or time_hit

            if should_exit:
                if stop_hit and tp_hit:
                    if intrabar_is_best:
                        stop_hit = False
                    else:
                        tp_hit = False

                # Calculate exit_price and R-multiple
                if stop_hit:
                    exit_price = float(sl_eff)
                    outcome = "SL" if exit_model != "ATR Trailing Stop" else "TRAIL_SL"
                elif tp_hit:
                    exit_price = float(tp)
                    outcome = "TP"
                else:
                    exit_price = float(row["close"])
                    outcome = "EMA_FLIP"

                if time_hit and not (stop_hit or tp_hit):
                    outcome = "TIME"
                if opp_hit and not (stop_hit or tp_hit or ema_flip_hit or time_hit):
                    outcome = "Opp"

                if pos == 1:
                    rdist = (entry - sl)
                    r_mult = (exit_price - entry) / rdist if rdist > 0 else 0.0
                else:
                    rdist = (sl - entry)
                    r_mult = (entry - exit_price) / rdist if rdist > 0 else 0.0

                pnl = float(r_mult) * float(risk_used)

                equity_before = float(equity)
                equity_after = float(equity + pnl)
                equity = equity_after

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
                        "risk_mode": str(risk_mode),
                        "risk_used": float(risk_used),
                        "qty": float(qty),
                        "notional": float(qty * float(entry)),
                        "p_win": float(signal_proba) if signal_proba is not None and np.isfinite(signal_proba) else np.nan,
                        "r_mult": float(r_mult),
                        "pnl": float(pnl),
                        "outcome": outcome,
                        "exit_model": exit_model,
                        "bars_in_trade": int(bars_in_trade),
                        "score": float(row.get("score", np.nan)),
                        "reason": entry_snapshot_json,
                        "mfe": float(mfe),
                        "mae": float(mae),
                        "entry_atr": float(entry_atr) if entry_atr is not None else np.nan,
                        "equity_before": equity_before,
                        "equity_after": equity_after,
                    }
                )

                # reset position
                pos = 0
                entry = sl = tp = None
                entry_time = None
                signal_time = None
                signal_proba = None
                signal_tag = None
                bars_in_trade = 0
                trail = None
                mfe = mae = 0.0
                entry_atr = None
                entry_snapshot_json = None
                qty = 0.0
                risk_used = 0.0

    return d, pd.DataFrame(trades)


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    df_view: pd.DataFrame,
    strategy_name: str,
    weights: dict[str, float],
    min_score: float,
    atr_len: int,
    atr_mult: float,
    rr: float,
    risk_mode: str,
    risk: float,
    start_equity: float,
    risk_pct: float,
    leverage_cap: float,
    min_notional: float,
    max_notional: float,
    both_dirs: bool,
    exit_on_opp: bool,
    fill_next_open: bool,
    intrabar_rule: str,
    exit_model: str,
    trail_atr_mult: float,
    time_stop_bars: int,
    use_ml: bool,
    proba,
    proba_thresh: float,
):
    return backtest(
        df_view,
        strategy_name=strategy_name,
        weights=weights,
        min_score=min_score,
        atr_len=atr_len,
        atr_mult=atr_mult,
        rr=rr,
        risk_mode=risk_mode,
        risk=risk,
        start_equity=start_equity,
        risk_pct=risk_pct,
        leverage_cap=leverage_cap,
        min_notional=min_notional,
        max_notional=max_notional,
        both_dirs=both_dirs,
        exit_on_opp=exit_on_opp,
        fill_next_open=fill_next_open,
        intrabar_rule=intrabar_rule,
        exit_model=exit_model,
        trail_atr_mult=trail_atr_mult,
        time_stop_bars=time_stop_bars,
        use_ml=use_ml,
        proba=proba,
        proba_thresh=proba_thresh,
    )


# ============================================================
# WALK-FORWARD ML
# ============================================================
def walk_forward_proba(
    d_sig: pd.DataFrame,
    X_all: np.ndarray,
    train_bars: int,
    test_bars: int,
    strategy_name: str,
    weights: dict[str, float],
    min_score: float,
    atr_len: int,
    atr_mult: float,
    rr: float,
    risk_mode: str,
    risk: float,
    start_equity: float,
    risk_pct: float,
    leverage_cap: float,
    min_notional: float,
    max_notional: float,
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

        # label using backtest outcomes
        _, tt = backtest(
            d_train,
            strategy_name=strategy_name,
            weights=weights,
            min_score=min_score,
            atr_len=atr_len,
            atr_mult=atr_mult,
            rr=rr,
            risk_mode=risk_mode,
            risk=risk,
            start_equity=start_equity,
            risk_pct=risk_pct,
            leverage_cap=leverage_cap,
            min_notional=min_notional,
            max_notional=max_notional,
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


# ============================================================
# SESSION OVERLAYS
# ============================================================
def add_session_overlays(fig, dts: pd.Series, show_asia=True, show_london=True, show_ny=True):
    """
    UTC session boxes per day.
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


# ============================================================
# CHART TRANSFORMS
# ============================================================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ha_close = (d["open"] + d["high"] + d["low"] + d["close"]) / 4.0
    ha_open = np.zeros(len(d))
    ha_open[0] = (d["open"].iloc[0] + d["close"].iloc[0]) / 2.0
    for i in range(1, len(d)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = np.maximum.reduce([d["high"].to_numpy(), ha_open, ha_close.to_numpy()])
    ha_low = np.minimum.reduce([d["low"].to_numpy(), ha_open, ha_close.to_numpy()])
    d["open"] = ha_open
    d["high"] = ha_high
    d["low"] = ha_low
    d["close"] = ha_close
    return d


# ============================================================
# PRESETS + SETTINGS SAVE/LOAD
# ============================================================
@dataclass
class Preset:
    name: str
    values: dict[str, Any]


DEFAULT_PRESETS = [
    Preset("Trend (Crossover)", {"strategy_name": "MA Crossover", "timeframe": "15m", "fast": 10, "slow": 20, "trend_len": 200, "min_score": 55.0}),
    Preset("Momentum", {"strategy_name": "Momentum (ROC+VOL+RSI)", "timeframe": "15m", "roc_len": 20, "momentum_roc_thresh": 0.35, "min_score": 58.0}),
    Preset("Mean Reversion (RSI)", {"strategy_name": "RSI Mean Reversion", "timeframe": "15m", "rsi_len": 14, "min_score": 52.0}),
    Preset("Breakout (Range)", {"strategy_name": "Range Breakout", "timeframe": "15m", "range_len": 60, "break_close": True, "min_score": 56.0}),
    Preset("Squeeze (BB)", {"strategy_name": "Bollinger Squeeze Breakout", "timeframe": "15m", "bb_len": 20, "bb_k": 2.0, "squeeze_pct": 2.0, "min_score": 56.0}),
]


def get_settings() -> dict[str, Any]:
    keys = [
        "timeframe",
        "strategy_name",
        "fast",
        "slow",
        "trend_len",
        "rsi_len",
        "roc_len",
        "momentum_roc_thresh",
        "vol_len",
        "range_len",
        "break_close",
        "bb_len",
        "bb_k",
        "squeeze_pct",
        "use_session_vwap",
        "vwap_dev",
        "min_score",
        "w_trend",
        "w_rsi",
        "w_vol",
        "w_strength",
        "w_compression",
        "w_pullback",
        "w_roc",
        "w_wick",
        "w_dev",
        "atr_len",
        "atr_mult",
        "rr",
        "risk_mode",
        "risk",
        "start_equity",
        "risk_pct",
        "leverage_cap",
        "min_notional",
        "max_notional",
        "both_dirs",
        "exit_on_opp",
        "fill_next_open",
        "intrabar_rule",
        "exit_override",
        "trail_atr_mult",
        "time_stop_bars",
        "use_ml",
        "ml_mode",
        "train_frac",
        "wf_train_bars",
        "wf_test_bars",
        "proba_thresh",
        "chart_style",
        "chart_last_n",
        "show_indicators",
        "show_sessions",
        "show_asia",
        "show_london",
        "show_ny",
        "marker_mode",
        "max_markers",
        "show_selected_sl_tp",
        "show_ml_overlay",
    ]
    out = {}
    for k in keys:
        if k in st.session_state:
            out[k] = st.session_state[k]
    return out


def apply_settings(values: dict[str, Any]):
    for k, v in values.items():
        st.session_state[k] = v


# ============================================================
# UI
# ============================================================
st.title("BTC Strategy Trade Visualiser")

with st.sidebar:
    st.header("Upload Data")
    files = st.file_uploader("Binance BTCUSDT ZIP / CSV", type=["zip", "csv"], accept_multiple_files=True)

    cols = st.columns(2)
    with cols[0]:
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.rerun()
    with cols[1]:
        st.write("")

    st.divider()
    st.subheader("Presets & Settings")
    preset_names = ["(none)"] + [p.name for p in DEFAULT_PRESETS]
    preset_pick = st.selectbox("Apply preset", preset_names, index=0)
    if preset_pick != "(none)":
        p = next(pp for pp in DEFAULT_PRESETS if pp.name == preset_pick)
        apply_settings(p.values)
        st.success(f"Applied preset: {p.name}")

    s = get_settings()
    st.download_button("Download settings JSON", data=json.dumps(s, indent=2).encode("utf-8"), file_name="btc_visualiser_settings.json", mime="application/json")
    up = st.file_uploader("Load settings JSON", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            vals = json.loads(up.getvalue().decode("utf-8"))
            if isinstance(vals, dict):
                apply_settings(vals)
                st.success("Loaded settings.")
        except Exception:
            st.error("Could not parse settings JSON.")

    st.divider()

    tf_label_to_rule = {"1m": "1T", "5m": "5T", "10m": "10T", "15m": "15T", "1h": "1H", "4h": "4H"}
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = "15m"
    if "strategy_name" not in st.session_state:
        st.session_state["strategy_name"] = STRATEGIES[0]

    timeframe = st.selectbox("Timeframe", list(tf_label_to_rule.keys()), index=list(tf_label_to_rule.keys()).index(st.session_state["timeframe"]))
    st.session_state["timeframe"] = timeframe

    strategy_name = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state["strategy_name"]))
    st.session_state["strategy_name"] = strategy_name

    st.divider()

    # Organised controls
    with st.expander("Core Params", expanded=True):
        st.session_state["fast"] = st.number_input("Fast SMA", 2, 500, int(st.session_state.get("fast", 10)))
        st.session_state["slow"] = st.number_input("Slow SMA", 3, 1000, int(st.session_state.get("slow", 20)))
        st.session_state["trend_len"] = st.number_input("Trend SMA", 10, 2000, int(st.session_state.get("trend_len", 200)))
        st.session_state["rsi_len"] = st.number_input("RSI Length", 2, 100, int(st.session_state.get("rsi_len", 14)))
        st.session_state["roc_len"] = st.number_input("ROC Length", 2, 200, int(st.session_state.get("roc_len", 20)))
        st.session_state["momentum_roc_thresh"] = st.number_input("Momentum ROC thr (%)", 0.05, 5.0, float(st.session_state.get("momentum_roc_thresh", 0.25)), 0.05)
        st.session_state["vol_len"] = st.number_input("Volume MA Length", 2, 500, int(st.session_state.get("vol_len", 20)))
        st.session_state["range_len"] = st.number_input("Range lookback", 5, 500, int(st.session_state.get("range_len", 50)))
        st.session_state["break_close"] = st.toggle("Range breakout requires CLOSE outside", bool(st.session_state.get("break_close", True)))
        st.session_state["bb_len"] = st.number_input("BB Length", 5, 300, int(st.session_state.get("bb_len", 20)))
        st.session_state["bb_k"] = st.number_input("BB StdDev (K)", 1.0, 4.0, float(st.session_state.get("bb_k", 2.0)), 0.25)
        st.session_state["squeeze_pct"] = st.number_input("Squeeze threshold (BB width %)", 0.1, 20.0, float(st.session_state.get("squeeze_pct", 2.0)), 0.1)
        st.session_state["use_session_vwap"] = st.toggle("VWAP daily reset", bool(st.session_state.get("use_session_vwap", True)))
        st.session_state["vwap_dev"] = st.number_input("VWAP deviation trigger (%)", 0.1, 10.0, float(st.session_state.get("vwap_dev", 1.0)), 0.1)

    with st.expander("Confluence Scoring", expanded=False):
        st.session_state["min_score"] = st.slider("Min score", 0.0, 100.0, float(st.session_state.get("min_score", 55.0)), 1.0)
        st.session_state["w_trend"] = st.slider("Weight: Trend", 0.0, 5.0, float(st.session_state.get("w_trend", 2.0)), 0.5)
        st.session_state["w_rsi"] = st.slider("Weight: RSI", 0.0, 5.0, float(st.session_state.get("w_rsi", 1.5)), 0.5)
        st.session_state["w_vol"] = st.slider("Weight: Volume", 0.0, 5.0, float(st.session_state.get("w_vol", 1.5)), 0.5)
        st.session_state["w_strength"] = st.slider("Weight: Strength/Distance", 0.0, 5.0, float(st.session_state.get("w_strength", 1.5)), 0.5)
        st.session_state["w_compression"] = st.slider("Weight: Compression", 0.0, 5.0, float(st.session_state.get("w_compression", 1.0)), 0.5)
        st.session_state["w_pullback"] = st.slider("Weight: Pullback", 0.0, 5.0, float(st.session_state.get("w_pullback", 1.0)), 0.5)
        st.session_state["w_roc"] = st.slider("Weight: ROC", 0.0, 5.0, float(st.session_state.get("w_roc", 2.0)), 0.5)
        st.session_state["w_wick"] = st.slider("Weight: Wick/Rejection", 0.0, 5.0, float(st.session_state.get("w_wick", 1.0)), 0.5)
        st.session_state["w_dev"] = st.slider("Weight: Deviation", 0.0, 5.0, float(st.session_state.get("w_dev", 2.0)), 0.5)

    with st.expander("Risk / Execution / Sizing", expanded=False):
        st.session_state["atr_len"] = st.number_input("ATR Length", 2, 100, int(st.session_state.get("atr_len", 14)))
        st.session_state["atr_mult"] = st.number_input("ATR Multiplier", 0.5, 10.0, float(st.session_state.get("atr_mult", 2.5)))
        st.session_state["rr"] = st.number_input("Risk Reward (RR)", 0.5, 10.0, float(st.session_state.get("rr", 2.0)))

        st.session_state["risk_mode"] = st.selectbox("Risk mode", ["Fixed $", "% Equity"], index=0 if st.session_state.get("risk_mode", "Fixed $") == "Fixed $" else 1)
        st.session_state["risk"] = st.number_input("Fixed risk per trade ($)", 1.0, 1_000_000.0, float(st.session_state.get("risk", 100.0)))
        st.session_state["start_equity"] = st.number_input("Starting equity ($)", 1.0, 100_000_000.0, float(st.session_state.get("start_equity", 10_000.0)))
        st.session_state["risk_pct"] = st.number_input("Risk % of equity", 0.01, 20.0, float(st.session_state.get("risk_pct", 1.0)), 0.1)

        st.session_state["leverage_cap"] = st.number_input("Leverage cap (x)", 0.1, 200.0, float(st.session_state.get("leverage_cap", 5.0)), 0.5)
        st.session_state["min_notional"] = st.number_input("Min position notional ($)", 0.0, 100_000_000.0, float(st.session_state.get("min_notional", 0.0)))
        st.session_state["max_notional"] = st.number_input("Max position notional ($)", 0.0, 100_000_000.0, float(st.session_state.get("max_notional", 0.0)))

        st.session_state["both_dirs"] = st.toggle("Trade Long & Short", bool(st.session_state.get("both_dirs", True)))
        st.session_state["exit_on_opp"] = st.toggle("Exit on opposite signal (extra)", bool(st.session_state.get("exit_on_opp", False)))
        st.session_state["fill_next_open"] = st.toggle("Enter next candle OPEN", bool(st.session_state.get("fill_next_open", True)))
        st.session_state["intrabar_rule"] = st.selectbox("If TP & SL hit same candle", ["Worst case (SL first)", "Best case (TP first)"], index=0)

    with st.expander("Exit Model", expanded=False):
        st.session_state["exit_override"] = st.selectbox("Exit model", EXIT_MODELS, index=EXIT_MODELS.index(st.session_state.get("exit_override", "AUTO")) if st.session_state.get("exit_override", "AUTO") in EXIT_MODELS else 0)
        st.session_state["trail_atr_mult"] = st.number_input("Trailing ATR multiple", 0.5, 10.0, float(st.session_state.get("trail_atr_mult", 2.0)), 0.25)
        st.session_state["time_stop_bars"] = st.number_input("Time stop (bars)", 5, 5000, int(st.session_state.get("time_stop_bars", 48)))

    with st.expander("ML Filter", expanded=False):
        st.session_state["use_ml"] = st.toggle("Enable ML filter", bool(st.session_state.get("use_ml", False)))
        st.session_state["ml_mode"] = st.selectbox("ML mode", ["Single-split (fast)", "Walk-forward (realistic)"], index=1 if st.session_state.get("ml_mode", "Walk-forward (realistic)") == "Walk-forward (realistic)" else 0)
        st.session_state["train_frac"] = st.slider("Single-split train fraction", 0.3, 0.9, float(st.session_state.get("train_frac", 0.7)))
        st.session_state["wf_train_bars"] = st.number_input("Walk-forward train bars", 200, 20000, int(st.session_state.get("wf_train_bars", 3000)))
        st.session_state["wf_test_bars"] = st.number_input("Walk-forward test bars", 50, 5000, int(st.session_state.get("wf_test_bars", 500)))
        st.session_state["proba_thresh"] = st.slider("Only take trades if P(win) ≥", 0.50, 0.80, float(st.session_state.get("proba_thresh", 0.55)), 0.01)

    with st.expander("Chart (Lag controls)", expanded=False):
        st.session_state["chart_style"] = st.selectbox("Chart style", ["Candles", "Heikin Ashi", "OHLC", "Close line (fast)", "Close area (fast)"], index=["Candles", "Heikin Ashi", "OHLC", "Close line (fast)", "Close area (fast)"].index(st.session_state.get("chart_style", "Candles")) if st.session_state.get("chart_style", "Candles") in ["Candles", "Heikin Ashi", "OHLC", "Close line (fast)", "Close area (fast)"] else 0)
        st.session_state["chart_last_n"] = st.number_input("Chart last N candles", 200, 200_000, int(st.session_state.get("chart_last_n", 4000)))
        st.session_state["show_indicators"] = st.toggle("Show indicators", bool(st.session_state.get("show_indicators", True)))

        st.session_state["show_sessions"] = st.toggle("Show session overlays", bool(st.session_state.get("show_sessions", False)))
        st.session_state["show_asia"] = st.toggle("Asia (00–06)", bool(st.session_state.get("show_asia", True)))
        st.session_state["show_london"] = st.toggle("London (07–10)", bool(st.session_state.get("show_london", True)))
        st.session_state["show_ny"] = st.toggle("NY (13–16)", bool(st.session_state.get("show_ny", True)))

        st.session_state["marker_mode"] = st.selectbox("Trade markers", ["Selected trade only (fast)", "All trades (can lag)"], index=0 if st.session_state.get("marker_mode", "Selected trade only (fast)").startswith("Selected") else 1)
        st.session_state["max_markers"] = st.number_input("Max markers (when showing all)", 0, 20000, int(st.session_state.get("max_markers", 2000)))
        st.session_state["show_selected_sl_tp"] = st.toggle("Show SL/TP for selected trade", bool(st.session_state.get("show_selected_sl_tp", True)))
        st.session_state["show_ml_overlay"] = st.toggle("Show ML P(win) markers", bool(st.session_state.get("show_ml_overlay", True)))

    st.divider()
    run_now = st.button("Run backtest")


# ============================================================
# MAIN FLOW
# ============================================================
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

target_rule = {"1m": "1T", "5m": "5T", "10m": "10T", "15m": "15T", "1h": "1H", "4h": "4H"}[timeframe]
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

# Auto-run once
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if not run_now and st.session_state.has_run is False:
    run_now = True
if not run_now:
    st.info("Adjust settings, then click Run backtest.")
    st.stop()
st.session_state.has_run = True

exit_model = st.session_state["exit_override"]
if exit_model == "AUTO":
    exit_model = DEFAULT_EXIT_FOR_STRAT.get(strategy_name, "TP/SL only")

weights = {
    "trend": float(st.session_state.get("w_trend", 2.0)),
    "rsi": float(st.session_state.get("w_rsi", 1.5)),
    "vol": float(st.session_state.get("w_vol", 1.5)),
    "strength": float(st.session_state.get("w_strength", 1.5)),
    "compression": float(st.session_state.get("w_compression", 1.0)),
    "pullback": float(st.session_state.get("w_pullback", 1.0)),
    "roc": float(st.session_state.get("w_roc", 2.0)),
    "wick": float(st.session_state.get("w_wick", 1.0)),
    "dev": float(st.session_state.get("w_dev", 2.0)),
    "ma_sep": float(st.session_state.get("w_strength", 1.5)),
    "reclaim": float(st.session_state.get("w_strength", 1.5)),
}

d_sig = compute_signals(
    df_tf,
    strategy_name=strategy_name,
    fast=int(st.session_state["fast"]),
    slow=int(st.session_state["slow"]),
    trend_len=int(st.session_state["trend_len"]),
    rsi_len=int(st.session_state["rsi_len"]),
    roc_len=int(st.session_state["roc_len"]),
    vol_len=int(st.session_state["vol_len"]),
    range_len=int(st.session_state["range_len"]),
    break_close=bool(st.session_state["break_close"]),
    bb_len=int(st.session_state["bb_len"]),
    bb_k=float(st.session_state["bb_k"]),
    squeeze_pct=float(st.session_state["squeeze_pct"]),
    use_session_vwap=bool(st.session_state["use_session_vwap"]),
    vwap_dev=float(st.session_state["vwap_dev"]),
    momentum_roc_thresh=float(st.session_state["momentum_roc_thresh"]),
    weights=weights,
    min_score=float(st.session_state["min_score"]),
)

if d_sig is None or (not isinstance(d_sig, pd.DataFrame)) or d_sig.empty:
    st.error("Signals dataframe is invalid/empty. Check your data, date range, and timeframe.")
    st.stop()

# ML
proba = None
ml_info = None
use_ml = bool(st.session_state.get("use_ml", False))

if use_ml:
    X_all, feat_cols = build_ml_features(d_sig)

    if st.session_state.get("ml_mode", "Walk-forward (realistic)") == "Walk-forward (realistic)":
        proba, nwin, ntr = walk_forward_proba(
            d_sig,
            X_all,
            train_bars=int(st.session_state.get("wf_train_bars", 3000)),
            test_bars=int(st.session_state.get("wf_test_bars", 500)),
            strategy_name=strategy_name,
            weights=weights,
            min_score=float(st.session_state["min_score"]),
            atr_len=int(st.session_state["atr_len"]),
            atr_mult=float(st.session_state["atr_mult"]),
            rr=float(st.session_state["rr"]),
            risk_mode=str(st.session_state["risk_mode"]),
            risk=float(st.session_state["risk"]),
            start_equity=float(st.session_state["start_equity"]),
            risk_pct=float(st.session_state["risk_pct"]),
            leverage_cap=float(st.session_state["leverage_cap"]),
            min_notional=float(st.session_state["min_notional"]),
            max_notional=float(st.session_state["max_notional"]),
            both_dirs=bool(st.session_state["both_dirs"]),
            exit_on_opp=bool(st.session_state["exit_on_opp"]),
            fill_next_open=bool(st.session_state["fill_next_open"]),
            intrabar_rule=str(st.session_state["intrabar_rule"]),
            exit_model=str(exit_model),
            trail_atr_mult=float(st.session_state["trail_atr_mult"]),
            time_stop_bars=int(st.session_state["time_stop_bars"]),
        )
        ml_info = f"Walk-forward ML: windows trained={nwin}, labeled trades used={ntr}. Features: {', '.join(feat_cols)}"
    else:
        cut_idx = int(len(d_sig) * float(st.session_state.get("train_frac", 0.7)))
        cut_idx = max(50, min(cut_idx, len(d_sig) - 50))
        train_end_dt = d_sig["dt"].iloc[cut_idx]
        d_train = d_sig[d_sig["dt"] <= train_end_dt].copy()

        _, tt = backtest(
            d_train,
            strategy_name=strategy_name,
            weights=weights,
            min_score=float(st.session_state["min_score"]),
            atr_len=int(st.session_state["atr_len"]),
            atr_mult=float(st.session_state["atr_mult"]),
            rr=float(st.session_state["rr"]),
            risk_mode=str(st.session_state["risk_mode"]),
            risk=float(st.session_state["risk"]),
            start_equity=float(st.session_state["start_equity"]),
            risk_pct=float(st.session_state["risk_pct"]),
            leverage_cap=float(st.session_state["leverage_cap"]),
            min_notional=float(st.session_state["min_notional"]),
            max_notional=float(st.session_state["max_notional"]),
            both_dirs=bool(st.session_state["both_dirs"]),
            exit_on_opp=bool(st.session_state["exit_on_opp"]),
            fill_next_open=bool(st.session_state["fill_next_open"]),
            intrabar_rule=str(st.session_state["intrabar_rule"]),
            exit_model=str(exit_model),
            trail_atr_mult=float(st.session_state["trail_atr_mult"]),
            time_stop_bars=int(st.session_state["time_stop_bars"]),
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
                st.warning("ML: Training failed. Running without ML filter.")
                use_ml = False
                proba = None
            else:
                proba = predict_logreg_proba(X_all, model)
                ml_info = f"Single-split ML trained ≤ {pd.Timestamp(train_end_dt):%Y-%m-%d}. Features: {', '.join(feat_cols)}"

if ml_info:
    st.info(ml_info)

d_full, trades = run_backtest_cached(
    d_sig,
    strategy_name=strategy_name,
    weights=weights,
    min_score=float(st.session_state["min_score"]),
    atr_len=int(st.session_state["atr_len"]),
    atr_mult=float(st.session_state["atr_mult"]),
    rr=float(st.session_state["rr"]),
    risk_mode=str(st.session_state["risk_mode"]),
    risk=float(st.session_state["risk"]),
    start_equity=float(st.session_state["start_equity"]),
    risk_pct=float(st.session_state["risk_pct"]),
    leverage_cap=float(st.session_state["leverage_cap"]),
    min_notional=float(st.session_state["min_notional"]),
    max_notional=float(st.session_state["max_notional"]),
    both_dirs=bool(st.session_state["both_dirs"]),
    exit_on_opp=bool(st.session_state["exit_on_opp"]),
    fill_next_open=bool(st.session_state["fill_next_open"]),
    intrabar_rule=str(st.session_state["intrabar_rule"]),
    exit_model=str(exit_model),
    trail_atr_mult=float(st.session_state["trail_atr_mult"]),
    time_stop_bars=int(st.session_state["time_stop_bars"]),
    use_ml=bool(use_ml),
    proba=proba,
    proba_thresh=float(st.session_state["proba_thresh"]),
)

# Metrics
wins = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
losses = int((trades["pnl"] < 0).sum()) if not trades.empty else 0
net = float(trades["pnl"].sum()) if not trades.empty else 0.0
wr = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
avg_r = float(trades["r_mult"].mean()) if (not trades.empty and "r_mult" in trades.columns) else 0.0
avg_p = float(trades["p_win"].mean()) if (not trades.empty and "p_win" in trades.columns) else np.nan
avg_score = float(trades["score"].mean()) if (not trades.empty and "score" in trades.columns) else np.nan
end_equity = float(trades["equity_after"].iloc[-1]) if (not trades.empty and "equity_after" in trades.columns) else float(st.session_state["start_equity"])

c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
c1.metric("Trades", len(trades))
c2.metric("Win rate", f"{wr:.1f}%")
c3.metric("Wins", wins)
c4.metric("Losses", losses)
c5.metric("Net PnL ($)", f"{net:.0f}")
c6.metric("Avg R", f"{avg_r:.2f}")
c7.metric("Avg P(win)", f"{avg_p:.2f}" if np.isfinite(avg_p) else "—")
c8.metric("Avg Score", f"{avg_score:.1f}" if np.isfinite(avg_score) else "—")
c9.metric("End Equity", f"{end_equity:,.0f}")

tabs = st.tabs(["Chart", "Trades Table", "Trade Drilldown"])

# ----------------------------
# Chart tab
# ----------------------------
with tabs[0]:
    d_chart = d_full.copy()
    last_n = int(st.session_state.get("chart_last_n", 4000))
    if last_n > 0 and len(d_chart) > last_n:
        d_chart = d_chart.iloc[-last_n:].copy()

    # chart style transform
    chart_style = st.session_state.get("chart_style", "Candles")
    plot_df = d_chart.copy()
    if chart_style == "Heikin Ashi":
        plot_df = heikin_ashi(plot_df)

    fig = go.Figure()

    if chart_style in ["Candles", "Heikin Ashi"]:
        fig.add_trace(
            go.Candlestick(
                x=plot_df["dt"],
                open=plot_df["open"],
                high=plot_df["high"],
                low=plot_df["low"],
                close=plot_df["close"],
                name=f"BTCUSDT ({timeframe})",
            )
        )
    elif chart_style == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=plot_df["dt"],
                open=plot_df["open"],
                high=plot_df["high"],
                low=plot_df["low"],
                close=plot_df["close"],
                name=f"BTCUSDT ({timeframe})",
            )
        )
    elif chart_style == "Close area (fast)":
        fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["close"], mode="lines", fill="tozeroy", name="Close"))
    else:
        fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["close"], mode="lines", name="Close"))

    # sessions (auto limit)
    if bool(st.session_state.get("show_sessions", False)):
        days_span = (plot_df["dt"].max() - plot_df["dt"].min()).days if len(plot_df) else 0
        if days_span > 60:
            st.warning("Session overlays disabled automatically for large chart spans (>60 days). Reduce chart range or last N.")
        else:
            fig = add_session_overlays(
                fig,
                plot_df["dt"],
                show_asia=bool(st.session_state.get("show_asia", True)),
                show_london=bool(st.session_state.get("show_london", True)),
                show_ny=bool(st.session_state.get("show_ny", True)),
            )

    # indicators (simple)
    if bool(st.session_state.get("show_indicators", True)):
        for col_name, label in [
            ("FAST", f"SMA {st.session_state['fast']}"),
            ("SLOW", f"SMA {st.session_state['slow']}"),
            ("TREND", f"SMA {st.session_state['trend_len']}"),
            ("EMA20", "EMA20"),
            ("EMA50", "EMA50"),
            ("BB_U", "BB Upper"),
            ("BB_M", "BB Mid"),
            ("BB_L", "BB Lower"),
            ("VWAP", "VWAP"),
            ("R_H", "Range High"),
            ("R_L", "Range Low"),
        ]:
            if col_name in plot_df.columns:
                fig.add_trace(go.Scattergl(x=plot_df["dt"], y=plot_df[col_name], mode="lines", name=label))

    # trade selector
    selected_idx = None
    t = None
    if not trades.empty:
        t = trades.reset_index(drop=True)
        labels = [
            f"#{i+1} {r.side} | {pd.to_datetime(r.entry_time):%Y-%m-%d %H:%M} → {pd.to_datetime(r.exit_time):%Y-%m-%d %H:%M} | {r.outcome} | "
            f"Score {r.score:.0f} | P(win) {r.p_win:.2f} | R {r.r_mult:+.2f} | {r.sig_tag} | {r.exit_model}"
            for i, r in t.iterrows()
        ]
        selected = st.selectbox("Select a trade to highlight", ["(none)"] + labels, index=0)
        if selected != "(none)":
            selected_idx = labels.index(selected)

    # markers
    if t is not None and not t.empty:
        mode = st.session_state.get("marker_mode", "Selected trade only (fast)")
        if mode.startswith("All"):
            max_m = int(st.session_state.get("max_markers", 2000))
            tt = t.copy()
            if max_m > 0 and len(tt) > max_m:
                tt = tt.iloc[-max_m:].copy()
            fig.add_trace(go.Scatter(x=tt["entry_time"], y=tt["entry_price"], mode="markers", name="Entries", marker=dict(symbol="triangle-up", size=8)))
            fig.add_trace(go.Scatter(x=tt["exit_time"], y=tt["exit_price"], mode="markers", name="Exits", marker=dict(symbol="x", size=8)))
        elif selected_idx is not None:
            r = t.iloc[selected_idx]
            fig.add_trace(
                go.Scatter(
                    x=[r["entry_time"], r["exit_time"]],
                    y=[r["entry_price"], r["exit_price"]],
                    mode="lines+markers",
                    name="Selected trade",
                    line=dict(width=4),
                    marker=dict(size=10),
                )
            )
            if bool(st.session_state.get("show_selected_sl_tp", True)):
                fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["sl"], r["sl"]], mode="lines", name="Selected SL"))
                fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["tp"], r["tp"]], mode="lines", name="Selected TP"))

    # ML overlay (signals)
    if bool(st.session_state.get("show_ml_overlay", True)) and use_ml and (proba is not None) and ("long_sig" in plot_df.columns):
        sig_mask = (plot_df["long_sig"] | plot_df["short_sig"])
        sig_points = plot_df[sig_mask].copy()
        if len(sig_points) > 0:
            idx = sig_points.index.to_numpy()
            p = np.array(proba)[idx]
            fig.add_trace(
                go.Scatter(
                    x=sig_points["dt"],
                    y=sig_points["close"],
                    mode="markers",
                    name="ML P(win) @ signals",
                    marker=dict(size=8, color=p, cmin=0.0, cmax=1.0, colorscale="Viridis", colorbar=dict(title="P(win)"), symbol="circle", opacity=0.85),
                    text=[f"P(win)={pp:.2f}" if np.isfinite(pp) else "P(win)=NA" for pp in p],
                    hovertemplate="%{text}<br>%{x}<br>Close=%{y}<extra></extra>",
                )
            )

    fig.update_layout(height=720, dragmode="pan", xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False})

# ----------------------------
# Trades table tab
# ----------------------------
with tabs[1]:
    st.subheader("Trades Table")
    if trades.empty:
        st.write("No trades in this range with current settings.")
    else:
        tshow = trades.copy()
        for col in ["signal_time", "entry_time", "exit_time"]:
            tshow[col] = pd.to_datetime(tshow[col]).dt.strftime("%Y-%m-%d %H:%M")
        # hide heavy reason column by default, but keep it downloadable
        cols = [c for c in tshow.columns if c != "reason"]
        st.dataframe(tshow[cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download trades CSV",
            data=trades.to_csv(index=False).encode("utf-8"),
            file_name=f"trades_{timeframe}_{strategy_name.replace(' ', '_')}.csv",
            mime="text/csv",
        )

# ----------------------------
# Drilldown tab
# ----------------------------
with tabs[2]:
    st.subheader("Trade Drilldown")
    if trades.empty:
        st.write("No trades to inspect.")
    else:
        t = trades.reset_index(drop=True)
        idx = st.number_input("Trade #", 1, len(t), 1) - 1
        tr = t.iloc[idx]

        reason = {}
        try:
            reason = json.loads(tr.get("reason", "{}")) if pd.notna(tr.get("reason", np.nan)) else {}
        except Exception:
            reason = {}

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Entry reasoning")
            st.write(f"**Strategy:** {reason.get('strategy','')}")
            st.write(f"**Signal tag:** {reason.get('sig_tag','')}")
            st.write(f"**Signal time:** {reason.get('signal_time','')}")
            st.write(f"**Score:** {reason.get('score', '—')}  |  **Min required:** {reason.get('min_score','—')}")
            st.write(f"**Raw trigger:** long={reason.get('raw_long', False)} short={reason.get('raw_short', False)}")

            ind = reason.get("indicators", {})
            st.write("**Indicators at signal candle:**")
            st.write(
                f"- RSI: {ind.get('RSI','—')}\n"
                f"- ROC: {ind.get('ROC','—')}\n"
                f"- VOL expansion: {ind.get('VOL_EXP','—')}\n"
                f"- ATR: {ind.get('ATR','—')}"
            )

            sz = reason.get("sizing", {})
            if sz:
                st.write("**Sizing at entry:**")
                st.write(
                    f"- Mode: {sz.get('risk_mode','')}\n"
                    f"- Equity: {sz.get('equity','—')}\n"
                    f"- Risk used: {sz.get('risk_used','—')}\n"
                    f"- Qty: {sz.get('qty','—')}\n"
                    f"- Notional: {sz.get('notional','—')}"
                )

        with col2:
            st.markdown("### Exit & outcome")
            st.write(f"**Exit model:** {tr.get('exit_model','')}")
            st.write(f"**Outcome:** {tr.get('outcome','')}")
            st.write(f"**R multiple:** {tr.get('r_mult','—'):.2f}" if pd.notna(tr.get("r_mult", np.nan)) else "—")
            st.write(f"**PnL ($):** {tr.get('pnl','—')}")
            st.write(f"**Bars held:** {int(tr.get('bars_in_trade',0))}")
            st.write(f"**MFE:** {tr.get('mfe','—'):.2f}  |  **MAE:** {tr.get('mae','—'):.2f}")
            st.write(f"**Equity:** {tr.get('equity_before','—'):.0f} → {tr.get('equity_after','—'):.0f}")

            st.markdown("**Why it won/lost (OHLC model):**")
            if str(tr.get("outcome", "")) == "TP":
                st.success("Price reached your take-profit before your stop (according to candle OHLC).")
            elif "SL" in str(tr.get("outcome", "")):
                st.error("Price hit your stop-loss (or trailing stop) before take-profit (according to candle OHLC).")
            elif str(tr.get("outcome", "")) == "EMA_FLIP":
                st.warning("EMA trend flipped against the position, so the exit model closed it.")
            elif str(tr.get("outcome", "")) == "TIME":
                st.warning("Time-stop closed the trade after the max holding period.")
            else:
                st.info("Closed due to opposite signal / exit condition.")

        st.markdown("### Confluence score breakdown")
        comps = reason.get("components", {})
        wts = reason.get("weights", {})
        contrib = reason.get("contribution", {})

        if comps:
            rows = []
            for k, v in comps.items():
                rows.append(
                    {
                        "component": k,
                        "value(0-1)": float(v) if v is not None else np.nan,
                        "weight": float(wts.get(k, 0.0)),
                        "weighted_contrib": float(contrib.get(k, 0.0)),
                    }
                )
            df_break = pd.DataFrame(rows).sort_values("weighted_contrib", ascending=False)
            st.dataframe(df_break, use_container_width=True, hide_index=True)
        else:
            st.write("No component breakdown available for this trade.")

