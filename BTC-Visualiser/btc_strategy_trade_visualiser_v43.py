# BTC Strategy Trade Visualiser (v5.1)
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
import json
import calendar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BTC Strategy Trade Visualiser (v5.1)", layout="wide", initial_sidebar_state="expanded")

# =====================
# UI THEME (light/dark + accent) — app-level CSS
# =====================
def apply_ui_theme(theme_mode: str, accent_hex: str):
    """Lightweight CSS theming + Plotly template selection."""
    theme_mode = (theme_mode or "Dark").strip().lower()
    accent_hex = accent_hex or "#00D1B2"

    if theme_mode.startswith("light"):
        bg = "#f6f7fb"
        panel = "#ffffff"
        text = "#111827"
        subtle = "#6b7280"
        border = "rgba(0,0,0,0.08)"
        plotly_template = "plotly_white"
    else:
        bg = "#0b1220"
        panel = "#0f172a"
        text = "#e5e7eb"
        subtle = "#9ca3af"
        border = "rgba(255,255,255,0.10)"
        plotly_template = "plotly_dark"

    css = f"""
    <style>
      .stApp {{
        background: {bg};
        color: {text};
      }}

      .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
      }}

      h1, h2, h3, h4, h5, h6, p, li, label {{
        color: {text};
      }}

      section[data-testid="stSidebar"] > div {{
        background: {panel};
        border-right: 1px solid {border};
      }}

      .stButton>button {{
        border-radius: 10px;
        border: 1px solid {border};
        background: {accent_hex};
        color: white;
        font-weight: 650;
      }}
      .stButton>button:hover {{
        filter: brightness(0.95);
        border-color: {accent_hex};
      }}

      div[data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 12px;
        overflow: hidden;
      }}

      .muted {{
        color: {subtle};
        font-size: 0.92rem;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return plotly_template

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
    # Store raw strategy triggers (before score filtering)
    x["raw_long"] = mask_long.astype(bool)
    x["raw_short"] = mask_short.astype(bool)

    parts_keys = list(parts_df.columns)

    # Save per-component normalized values (0..1)
    for k in parts_keys:
        x[f"score_{k}"] = parts_df[k].astype(float)

    # Vectorized score (0..100): weighted average of components in [0,1]
    w = np.array([float(weights.get(k, 0.0)) for k in parts_keys], dtype=float)
    vals = parts_df[parts_keys].to_numpy(dtype=float)
    vals = np.clip(vals, 0.0, 1.0)

    use = w > 0
    if np.any(use):
        ssum = vals[:, use] @ w[use]
        wsum = float(w[use].sum())
        x["score"] = (ssum / max(1e-12, wsum)) * 100.0
    else:
        x["score"] = 0.0

    ok = x["score"] >= float(min_score)
    x["long_sig"] = mask_long & ok
    x["short_sig"] = mask_short & ok
    x.loc[x["long_sig"], "sig_tag"] = tag_long
    x.loc[x["short_sig"], "sig_tag"] = tag_short

    # Save component list for later drill-down (same for all rows)
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
    # position sizing
    sizing_mode="Fixed $ risk",
    start_balance=10_000.0,
    risk_pct=1.0,
    leverage_cap=10.0,
    min_pos_usd=0.0,
    max_pos_usd=1_000_000.0,
    compound=True,
):
    d = df.copy()
    d["ATR"] = atr(d, atr_len)

    trades = []
    equity = float(start_balance)
    equity_before = np.nan
    qty = 0.0
    notional = 0.0
    risk_target = float(risk)
    risk_actual = np.nan
    leverage_used = np.nan
    size_capped = False
    min_forced = False

    pos = 0
    entry = sl = tp = None
    entry_time = None
    signal_time = None
    signal_proba = None
    signal_tag = None
    bars_in_trade = 0
    trail = None

    mfe = 0.0
    mae = 0.0
    entry_atr = None
    entry_score = None
    entry_snapshot_json = None

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
            entry_atr = None
            entry_score = None
            entry_snapshot_json = None
            qty = 0.0
            notional = 0.0
            risk_target = float(risk)
            risk_actual = np.nan
            leverage_used = np.nan
            equity_before = np.nan
            size_capped = False
            min_forced = False

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

                # Position sizing (qty in BTC, pnl in quote currency)
                equity_ref = float(equity) if bool(compound) else float(start_balance)
                equity_before = float(equity_ref)

                if str(sizing_mode).startswith("%"):
                    risk_target = float(equity_ref) * (float(risk_pct) / 100.0)
                else:
                    risk_target = float(risk)

                stop_dist = float(entry) - float(sl)
                if stop_dist <= 0:
                    # Invalid stop distance; skip this trade
                    pos = 0
                    entry = sl = tp = None
                    entry_time = None
                    signal_time = None
                    signal_proba = None
                    signal_tag = None
                    continue

                qty = (risk_target / stop_dist) if risk_target > 0 else 0.0
                notional = qty * float(entry)

                max_notional = float(max_pos_usd) if float(max_pos_usd) > 0 else float("inf")
                min_notional = float(min_pos_usd) if float(min_pos_usd) > 0 else 0.0
                max_lev_notional = float(equity_ref) * float(leverage_cap) if float(leverage_cap) > 0 else float("inf")
                notional_limit = min(max_notional, max_lev_notional)

                if notional > notional_limit:
                    qty = notional_limit / float(entry)
                    notional = qty * float(entry)
                    size_capped = True

                if min_notional > 0 and notional < min_notional:
                    qty_min = min_notional / float(entry)
                    qty_try = min(qty_min, notional_limit / float(entry))
                    if qty_try > qty:
                        qty = qty_try
                        notional = qty * float(entry)
                        min_forced = True

                risk_actual = float(qty) * float(stop_dist)
                leverage_used = (float(notional) / float(equity_ref)) if float(equity_ref) > 0 else np.nan

                # Snapshot at signal candle (row = signal candle)
                entry_atr = float(row["ATR"])
                entry_score = float(row.get("score", np.nan))

                # Build score breakdown
                comp_names = str(row.get("score_components", "")).split(",") if "score_components" in row else []
                comp_names = [c for c in comp_names if c]

                wts = globals().get("weights", {})
                wts = wts if isinstance(wts, dict) else {}

                components = {}
                contrib = {}
                for c in comp_names:
                    v = float(row.get(f"score_{c}", np.nan))
                    w = float(wts.get(c, 0.0))
                    components[c] = v
                    contrib[c] = (v * w)

                snapshot = {
                    "strategy": str(globals().get("strategy_name", "")),
                    "sig_tag": str(row.get("sig_tag", "")),
                    "signal_time": str(row["dt"]),
                    "entry_fill_mode": "next_open" if fill_next_open else "close",
                    "min_score": float(globals().get("min_score", np.nan)),
                    "score": entry_score,
                    "raw_long": bool(row.get("raw_long", False)),
                    "raw_short": bool(row.get("raw_short", False)),

                    "position_sizing": {
                        "mode": str(sizing_mode),
                        "equity_ref": float(equity_before) if pd.notna(equity_before) else float(start_balance),
                        "risk_target": float(risk_target) if pd.notna(risk_target) else np.nan,
                        "risk_actual": float(risk_actual) if pd.notna(risk_actual) else np.nan,
                        "qty": float(qty),
                        "notional": float(notional),
                        "leverage_used": float(leverage_used) if pd.notna(leverage_used) else np.nan,
                        "leverage_cap": float(leverage_cap),
                        "min_notional": float(min_pos_usd),
                        "max_notional": float(max_pos_usd),
                        "size_capped": bool(size_capped),
                        "min_forced": bool(min_forced),
                    },
                    "indicators": {
                        "RSI": float(row.get("RSI", np.nan)),
                        "ROC": float(row.get("ROC", np.nan)),
                        "VOL_EXP": bool(row.get("VOL_EXP", False)),
                        "ATR": float(row.get("ATR", np.nan)),
                    },
                    "components": components,
                    "weights": {k: float(v) for k, v in wts.items()},
                    "contribution": contrib,
                }

                entry_snapshot_json = json.dumps(snapshot, ensure_ascii=False)
                mfe = 0.0
                mae = 0.0

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

                # Position sizing (qty in BTC, pnl in quote currency)
                equity_ref = float(equity) if bool(compound) else float(start_balance)
                equity_before = float(equity_ref)

                if str(sizing_mode).startswith("%"):
                    risk_target = float(equity_ref) * (float(risk_pct) / 100.0)
                else:
                    risk_target = float(risk)

                stop_dist = float(sl) - float(entry)
                if stop_dist <= 0:
                    # Invalid stop distance; skip this trade
                    pos = 0
                    entry = sl = tp = None
                    entry_time = None
                    signal_time = None
                    signal_proba = None
                    signal_tag = None
                    continue

                qty = (risk_target / stop_dist) if risk_target > 0 else 0.0
                notional = qty * float(entry)

                max_notional = float(max_pos_usd) if float(max_pos_usd) > 0 else float("inf")
                min_notional = float(min_pos_usd) if float(min_pos_usd) > 0 else 0.0
                max_lev_notional = float(equity_ref) * float(leverage_cap) if float(leverage_cap) > 0 else float("inf")
                notional_limit = min(max_notional, max_lev_notional)

                if notional > notional_limit:
                    qty = notional_limit / float(entry)
                    notional = qty * float(entry)
                    size_capped = True

                if min_notional > 0 and notional < min_notional:
                    qty_min = min_notional / float(entry)
                    qty_try = min(qty_min, notional_limit / float(entry))
                    if qty_try > qty:
                        qty = qty_try
                        notional = qty * float(entry)
                        min_forced = True

                risk_actual = float(qty) * float(stop_dist)
                leverage_used = (float(notional) / float(equity_ref)) if float(equity_ref) > 0 else np.nan

                # Snapshot at signal candle (row = signal candle)
                entry_atr = float(row["ATR"])
                entry_score = float(row.get("score", np.nan))

                # Build score breakdown
                comp_names = str(row.get("score_components", "")).split(",") if "score_components" in row else []
                comp_names = [c for c in comp_names if c]

                wts = globals().get("weights", {})
                wts = wts if isinstance(wts, dict) else {}

                components = {}
                contrib = {}
                for c in comp_names:
                    v = float(row.get(f"score_{c}", np.nan))
                    w = float(wts.get(c, 0.0))
                    components[c] = v
                    contrib[c] = (v * w)

                snapshot = {
                    "strategy": str(globals().get("strategy_name", "")),
                    "sig_tag": str(row.get("sig_tag", "")),
                    "signal_time": str(row["dt"]),
                    "entry_fill_mode": "next_open" if fill_next_open else "close",
                    "min_score": float(globals().get("min_score", np.nan)),
                    "score": entry_score,
                    "raw_long": bool(row.get("raw_long", False)),
                    "raw_short": bool(row.get("raw_short", False)),

                    "position_sizing": {
                        "mode": str(sizing_mode),
                        "equity_ref": float(equity_before) if pd.notna(equity_before) else float(start_balance),
                        "risk_target": float(risk_target) if pd.notna(risk_target) else np.nan,
                        "risk_actual": float(risk_actual) if pd.notna(risk_actual) else np.nan,
                        "qty": float(qty),
                        "notional": float(notional),
                        "leverage_used": float(leverage_used) if pd.notna(leverage_used) else np.nan,
                        "leverage_cap": float(leverage_cap),
                        "min_notional": float(min_pos_usd),
                        "max_notional": float(max_pos_usd),
                        "size_capped": bool(size_capped),
                        "min_forced": bool(min_forced),
                    },
                    "indicators": {
                        "RSI": float(row.get("RSI", np.nan)),
                        "ROC": float(row.get("ROC", np.nan)),
                        "VOL_EXP": bool(row.get("VOL_EXP", False)),
                        "ATR": float(row.get("ATR", np.nan)),
                    },
                    "components": components,
                    "weights": {k: float(v) for k, v in wts.items()},
                    "contribution": contrib,
                }

                entry_snapshot_json = json.dumps(snapshot, ensure_ascii=False)
                mfe = 0.0
                mae = 0.0

        else:
            if fill_next_open and row["dt"] < entry_time:
                continue

            bars_in_trade += 1

# Track excursion from entry using candle extremes
            if pos == 1:
                # favourable = high - entry, adverse = entry - low
                mfe = max(mfe, float(row["high"]) - float(entry))
                mae = max(mae, float(entry) - float(row["low"]))
            else:
                # short: favourable = entry - low, adverse = high - entry
                mfe = max(mfe, float(entry) - float(row["low"]))
                mae = max(mae, float(row["high"]) - float(entry))

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


                # Real PnL ($) from position size (qty BTC)
                if pos == 1:
                    pnl = float(exit_price - entry) * float(qty)
                else:
                    pnl = float(entry - exit_price) * float(qty)

                equity_after = float(equity) + float(pnl)
                equity = float(equity_after)

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
                        "qty": float(qty),
                        "notional": float(notional),
                        "risk_target": float(risk_target) if pd.notna(risk_target) else np.nan,
                        "risk_actual": float(risk_actual) if pd.notna(risk_actual) else np.nan,
                        "equity_before": float(equity_before) if pd.notna(equity_before) else np.nan,
                        "equity_after": float(equity_after) if "equity_after" in locals() else np.nan,
                        "leverage_used": float(leverage_used) if pd.notna(leverage_used) else np.nan,
                        "size_capped": bool(size_capped),
                        "min_forced": bool(min_forced),
                        "outcome": outcome,
                        "exit_model": exit_model,
                        "bars_in_trade": int(bars_in_trade),
                        "score": float(row.get("score", np.nan)),
                        "reason": entry_snapshot_json,
                        "mfe": float(mfe),
                        "mae": float(mae),
                        "entry_atr": float(entry_atr) if entry_atr is not None else np.nan,
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
                mfe = 0.0
                mae = 0.0
                entry_atr = None
                entry_score = None
                entry_snapshot_json = None

    return d, pd.DataFrame(trades)


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    df,
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
    sizing_mode,
    start_balance,
    risk_pct,
    leverage_cap,
    min_pos_usd,
    max_pos_usd,
    compound,
):
    return backtest(
        df,
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
        sizing_mode=sizing_mode,
        start_balance=start_balance,
        risk_pct=risk_pct,
        leverage_cap=leverage_cap,
        min_pos_usd=min_pos_usd,
        max_pos_usd=max_pos_usd,
        compound=compound,
    )

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
# CHART TRANSFORMS
# =====================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin Ashi OHLC from standard OHLC. Intended for chart display."""
    if df.empty:
        return df
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    out = df.copy()
    out["open"] = ha_open
    out["high"] = ha_high
    out["low"] = ha_low
    out["close"] = ha_close
    return out


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
st.title("BTC Strategy Trade Visualiser (v5.1) — presets + sizing + chart styles + analytics")


# =====================
# SETTINGS PRESETS / SAVE-LOAD
# =====================
def _collect_ui_settings() -> dict:
    return {k: v for k, v in st.session_state.items() if k.startswith("ui_")}

def _apply_ui_settings(settings: dict, *, only_known: bool = True):
    if not isinstance(settings, dict):
        return
    for k, v in settings.items():
        if not str(k).startswith("ui_"):
            continue
        if only_known and (k not in st.session_state) and (k not in UI_KNOWN_KEYS):
            continue
        st.session_state[k] = v

# Keys we allow to be set from presets/files (prevents accidental junk)
UI_KNOWN_KEYS = {
    "ui_theme_mode", "ui_accent_hex",
    "ui_timeframe", "ui_strategy_name",
    "ui_fast", "ui_slow", "ui_trend_len",
    "ui_rsi_len", "ui_roc_len", "ui_momo_thr", "ui_vol_len", "ui_range_len", "ui_break_close",
    "ui_bb_len", "ui_bb_k", "ui_sqz", "ui_vwap_session", "ui_vwap_dev",
    "ui_min_score",
    "ui_w_trend", "ui_w_rsi", "ui_w_vol", "ui_w_strength", "ui_w_compress", "ui_w_pullback", "ui_w_roc", "ui_w_wick", "ui_w_dev",
    "ui_atr_len", "ui_atr_mult", "ui_rr",
    "ui_sizing_mode", "ui_start_balance", "ui_risk", "ui_risk_pct", "ui_leverage_cap", "ui_min_pos_usd", "ui_max_pos_usd", "ui_compound",
    "ui_both_dirs", "ui_exit_on_opp", "ui_fill_next_open", "ui_intrabar_rule",
    "ui_exit_override", "ui_trail_atr_mult", "ui_time_stop_bars",
    "ui_use_ml", "ui_ml_mode", "ui_train_frac", "ui_wf_train_bars", "ui_wf_test_bars", "ui_proba_thresh",
    "ui_chart_height", "ui_chart_window", "ui_chart_last_n", "ui_chart_cap",
    "ui_chart_style", "ui_show_indicators", "ui_show_ml_overlay",
    "ui_marker_mode", "ui_marker_limit", "ui_show_sl_tp",
    "ui_show_sessions", "ui_show_asia", "ui_show_london", "ui_show_ny", "ui_session_max_days", "ui_force_sessions",
}

PRESETS = {
    "Trend (smoother, fewer trades)": {
        "ui_strategy_name": "Trend Following (EMA pullback)",
        "ui_timeframe": "1h",
        "ui_min_score": 60.0,
        "ui_w_trend": 2.5,
        "ui_w_strength": 1.5,
        "ui_w_pullback": 1.5,
        "ui_w_rsi": 1.0,
        "ui_atr_mult": 2.8,
        "ui_rr": 2.0,
        "ui_chart_style": "Candles",
    },
    "Momentum (more signals)": {
        "ui_strategy_name": "Momentum (ROC+VOL+RSI)",
        "ui_timeframe": "15m",
        "ui_momo_thr": 0.25,
        "ui_min_score": 52.0,
        "ui_w_roc": 2.5,
        "ui_w_vol": 2.0,
        "ui_w_rsi": 1.0,
        "ui_atr_mult": 2.3,
        "ui_rr": 1.8,
        "ui_chart_style": "Close line (fast)",
    },
    "Mean reversion (RSI)": {
        "ui_strategy_name": "RSI Mean Reversion",
        "ui_timeframe": "15m",
        "ui_min_score": 50.0,
        "ui_w_rsi": 2.5,
        "ui_w_vol": 1.0,
        "ui_w_trend": 0.5,
        "ui_atr_mult": 2.2,
        "ui_rr": 1.5,
        "ui_exit_override": "Time Stop",
        "ui_time_stop_bars": 48,
        "ui_chart_style": "Candles",
    },
}

# Default template (overridden by sidebar theme)
plotly_template = "plotly_dark"


with st.sidebar:
    st.header("Controls")

    # ---- Appearance / UX ----
    with st.expander("🎨 Appearance & layout", expanded=False):
        theme_mode = st.selectbox("Theme", ["Dark", "Light"], index=0, key="ui_theme_mode", help="App UI theme (CSS + Plotly template).")
        accent_hex = st.color_picker("Accent color", "#00D1B2", key="ui_accent_hex")
        plotly_template = apply_ui_theme(theme_mode, accent_hex)

        compact_sidebar = st.toggle("Compact sidebar (use dropdown sections)", True)
        beginner_mode = st.toggle("Beginner mode (extra hints)", True)
        show_inline_help = st.toggle("Show inline parameter explanations", False, help="Turn ON for long per-parameter explainers (adds scrolling).")


    # ---- Presets / save / load ----
    with st.expander("💾 Presets & save/load", expanded=False):
        preset_names = ["(none)"] + list(PRESETS.keys())
        preset_pick = st.selectbox("Preset", preset_names, index=0, help="Apply a tested bundle of settings quickly.")
        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("Apply preset", use_container_width=True, disabled=(preset_pick == "(none)")):
                _apply_ui_settings(PRESETS.get(preset_pick, {}), only_known=True)
                st.rerun()
        with colp2:
            st.download_button(
                "Download settings JSON",
                data=json.dumps(_collect_ui_settings(), indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="btc_visualiser_settings.json",
                mime="application/json",
                use_container_width=True,
            )

        up = st.file_uploader("Load settings JSON", type=["json"], help="Upload a settings JSON previously downloaded here.")
        if up is not None:
            try:
                loaded = json.loads(up.getvalue().decode("utf-8"))
                _apply_ui_settings(loaded, only_known=True)
                st.success("Settings loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load JSON: {e}")

    # ---- Quick controls (favorites + search) ----
    with st.expander("⭐ Quick controls (favorites + search)", expanded=False):
        PARAM_DEFS = {
            "Min score": ("slider", "ui_min_score", dict(min_value=0.0, max_value=100.0, value=55.0, step=1.0)),
            "ATR mult": ("number", "ui_atr_mult", dict(min_value=0.5, max_value=10.0, value=2.5, step=0.1)),
            "RR": ("number", "ui_rr", dict(min_value=0.5, max_value=10.0, value=2.0, step=0.1)),
            "Fast SMA": ("number", "ui_fast", dict(min_value=2, max_value=500, value=10, step=1)),
            "Slow SMA": ("number", "ui_slow", dict(min_value=3, max_value=1000, value=20, step=1)),
            "Trend SMA": ("number", "ui_trend_len", dict(min_value=10, max_value=2000, value=200, step=5)),
            "Risk $ (fixed)": ("number", "ui_risk", dict(min_value=1.0, max_value=1_000_000.0, value=100.0, step=10.0)),
            "Risk % (equity)": ("number", "ui_risk_pct", dict(min_value=0.01, max_value=50.0, value=1.0, step=0.05)),
            "Leverage cap": ("number", "ui_leverage_cap", dict(min_value=1.0, max_value=200.0, value=10.0, step=1.0)),
            "Chart last N candles": ("number", "ui_chart_last_n", dict(min_value=300, max_value=100000, value=3500, step=100)),
        }

        fav_default = ["Min score", "ATR mult", "RR", "Chart last N candles"]
        favs = st.multiselect("Favorites", list(PARAM_DEFS.keys()), default=fav_default)

        q_search = st.text_input("Search settings (name contains...)")
        show_list = list(dict.fromkeys(favs + ([k for k in PARAM_DEFS.keys() if q_search and q_search.lower() in k.lower()])))

        if not show_list:
            st.caption("Pick favorites or search a parameter name to show quick controls.")
        else:
            for name in show_list:
                typ, master_key, cfg = PARAM_DEFS[name]
                # Ensure master exists so value(...) reads correctly
                if master_key not in st.session_state:
                    st.session_state[master_key] = cfg.get("value")
                cur = st.session_state.get(master_key, cfg.get("value"))
                k2 = f"qc_{master_key}"

                if typ == "slider":
                    val = st.slider(name, cfg["min_value"], cfg["max_value"], float(cur), float(cfg["step"]), key=k2)
                elif typ == "number":
                    # choose int vs float
                    if isinstance(cfg["min_value"], int) and isinstance(cfg["max_value"], int) and isinstance(cfg.get("step", 1), int):
                        val = st.number_input(name, int(cfg["min_value"]), int(cfg["max_value"]), int(cur), step=int(cfg.get("step", 1)), key=k2)
                    else:
                        val = st.number_input(name, float(cfg["min_value"]), float(cfg["max_value"]), float(cur), step=float(cfg.get("step", 0.1)), key=k2)
                else:
                    continue

                # Push back into master state (next rerun the main widget will reflect it)
                st.session_state[master_key] = val

    # ---- Data ----
    with st.expander("📁 Data upload", expanded=True):
        files = st.file_uploader("Binance BTCUSDT ZIP / CSV", type=["zip", "csv"], accept_multiple_files=True)
        colA, colB = st.columns(2)
        with colA:
            if st.button("Clear cache"):
                st.cache_data.clear()
                st.rerun()
        with colB:
            if st.button("Reset UI"):
                for k in list(st.session_state.keys()):
                    if k.startswith("ui_") or k in {"has_run"}:
                        st.session_state.pop(k, None)
                st.rerun()

    # ---- Help / glossary ----
    with st.expander("📚 Help / glossary", expanded=False):
        st.markdown('''
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
''')

    # ---- Timeframe ----
    with st.expander("⏱️ Timeframe", expanded=True if compact_sidebar else True):
        tf_label_to_rule = {"1m": "1T", "5m": "5T", "10m": "10T", "15m": "15T", "1h": "1H", "4h": "4H"}
        timeframe = st.selectbox(
            "Backtest timeframe",
            list(tf_label_to_rule.keys()),
            index=3,
            key="ui_timeframe",
            help="This changes BOTH the backtest timeframe and the default chart timeframe."
        )

    # ---- Strategy ----
    with st.expander("🧠 Strategy", expanded=True if compact_sidebar else True):
        strategy_name = st.selectbox("Choose strategy", STRATEGIES, index=0, key="ui_strategy_name")

        if beginner_mode:
            STRAT_EXPLAIN = {
                "MA Crossover": "Buys when fast SMA crosses above slow SMA; shorts on opposite. Best in trends.",
                "Momentum (ROC+VOL+RSI)": "Trades when price accelerates (ROC) with volume expansion and RSI confirmation.",
                "Trend Following (EMA pullback)": "Trades in trend direction after pullback; exits on trend flip.",
                "Range Breakout": "Trades breakout beyond recent range boundary.",
                "Bollinger Squeeze Breakout": "Trades volatility expansion after tight Bollinger Bands.",
                "RSI Mean Reversion": "Buys oversold / sells overbought aiming for snap-back.",
                "VWAP Reversion": "Trades when price deviates from VWAP expecting reversion to fair value.",
                "Liquidity Sweep (basic)": "Stop-hunt above/below equal highs/lows and trades the reversal.",
            }
            st.caption(STRAT_EXPLAIN.get(strategy_name, ""))

        st.markdown("**Core params**")
        fast = st.number_input("Fast SMA", 2, 500, 10, key="ui_fast")
        slow = st.number_input("Slow SMA", 3, 1000, 20, key="ui_slow")
        trend_len = st.number_input("Trend SMA", 10, 2000, 200, key="ui_trend_len")

        st.markdown("**Strategy params**")
        rsi_len = st.number_input("RSI Length", 2, 100, 14, key="ui_rsi_len")
        roc_len = st.number_input("ROC Length", 2, 200, 20, key="ui_roc_len")
        momentum_roc_thresh = st.number_input("Momentum ROC threshold (%)", 0.05, 5.0, 0.25, 0.05, key="ui_momo_thr")
        vol_len = st.number_input("Volume MA Length", 2, 500, 20, key="ui_vol_len")
        range_len = st.number_input("Range lookback", 5, 500, 50, key="ui_range_len")
        break_close = st.toggle("Range breakout requires CLOSE outside", True, key="ui_break_close")

        bb_len = st.number_input("BB Length", 5, 300, 20, key="ui_bb_len")
        bb_k = st.number_input("BB StdDev", 1.0, 4.0, 2.0, 0.25, key="ui_bb_k")
        squeeze_pct = st.number_input("Squeeze threshold (BB width %)", 0.1, 20.0, 2.0, 0.1, key="ui_sqz")

        use_session_vwap = st.toggle("VWAP: session reset (daily)", True, key="ui_vwap_session")
        vwap_dev = st.number_input("VWAP deviation trigger (%)", 0.1, 10.0, 1.0, 0.1, key="ui_vwap_dev")

        if show_inline_help and beginner_mode:
            st.info("Tip: Change ONE parameter at a time and re-test. Start with timeframe + date range first.")

    # ---- Confluence scoring ----
    with st.expander("🧩 Confluence scoring", expanded=False if compact_sidebar else True):
        min_score = st.slider("Min score to take trade", 0.0, 100.0, 55.0, 1.0, key="ui_min_score")

        col1, col2 = st.columns(2)
        with col1:
            w_trend = st.slider("Weight: Trend", 0.0, 5.0, 2.0, 0.5, key="ui_w_trend")
            w_rsi = st.slider("Weight: RSI", 0.0, 5.0, 1.5, 0.5, key="ui_w_rsi")
            w_vol = st.slider("Weight: Volume", 0.0, 5.0, 1.5, 0.5, key="ui_w_vol")
            w_strength = st.slider("Weight: Strength", 0.0, 5.0, 1.5, 0.5, key="ui_w_strength")
        with col2:
            w_compression = st.slider("Weight: Compression", 0.0, 5.0, 1.0, 0.5, key="ui_w_compress")
            w_pullback = st.slider("Weight: Pullback", 0.0, 5.0, 1.0, 0.5, key="ui_w_pullback")
            w_roc = st.slider("Weight: ROC", 0.0, 5.0, 2.0, 0.5, key="ui_w_roc")
            w_wick = st.slider("Weight: Wick/Rejection", 0.0, 5.0, 1.0, 0.5, key="ui_w_wick")
            w_dev = st.slider("Weight: Deviation", 0.0, 5.0, 2.0, 0.5, key="ui_w_dev")

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

    # ---- Risk / execution ----
    
    # ---- Risk / execution ----
    with st.expander("🛡️ Risk & execution", expanded=False if compact_sidebar else True):
        atr_len = st.number_input("ATR Length", 2, 100, 14, key="ui_atr_len")
        atr_mult = st.number_input("ATR Multiplier", 0.5, 10.0, 2.5, key="ui_atr_mult")
        rr = st.number_input("Risk Reward (RR)", 0.5, 10.0, 2.0, key="ui_rr")

        st.markdown("**Position sizing**")
        sizing_mode = st.selectbox(
            "Sizing mode",
            ["Fixed $ risk", "% of equity"],
            index=0,
            key="ui_sizing_mode",
            help="Fixed $ matches your old behaviour. % of equity compounds if enabled.",
        )
        start_balance = st.number_input("Starting balance ($)", 10.0, 1_000_000_000.0, 10_000.0, step=100.0, key="ui_start_balance")
        compound = st.toggle("Compound equity (size uses current balance)", True, key="ui_compound")

        if sizing_mode.startswith("Fixed"):
            risk = st.number_input("Risk per trade ($)", 1.0, 1_000_000.0, 100.0, key="ui_risk")
            risk_pct = st.number_input("Risk % per trade (info)", 0.01, 50.0, 1.0, step=0.05, key="ui_risk_pct", disabled=True)
        else:
            risk_pct = st.number_input("Risk % per trade", 0.01, 50.0, 1.0, step=0.05, key="ui_risk_pct")
            # keep a placeholder for compatibility; backtest sizes each trade from % equity
            risk = st.number_input("Risk per trade ($) (derived)", 1.0, 1_000_000.0, 100.0, key="ui_risk", disabled=True)

        leverage_cap = st.number_input("Leverage cap (x)", 1.0, 200.0, 10.0, step=1.0, key="ui_leverage_cap")
        min_pos_usd = st.number_input("Min position notional ($)", 0.0, 1_000_000_000.0, 0.0, step=10.0, key="ui_min_pos_usd")
        max_pos_usd = st.number_input("Max position notional ($)", 0.0, 1_000_000_000_000.0, 1_000_000.0, step=100.0, key="ui_max_pos_usd")
        if max_pos_usd > 0 and min_pos_usd > max_pos_usd:
            st.warning("Min notional is greater than max notional. Adjusting min down to max.")
            min_pos_usd = max_pos_usd
            st.session_state["ui_min_pos_usd"] = float(min_pos_usd)

        st.markdown("**Execution**")
        both_dirs = st.toggle("Trade Long & Short", True, key="ui_both_dirs")
        exit_on_opp = st.toggle("Exit on opposite signal (extra)", False, key="ui_exit_on_opp")

        fill_next_open = st.toggle("Enter on next candle OPEN (realistic)", True, key="ui_fill_next_open")
        intrabar_rule = st.selectbox(
            "If TP & SL hit same candle",
            ["Worst case (SL first)", "Best case (TP first)"],
            index=0,
            key="ui_intrabar_rule",
        )


    # ---- Exit model ----
    with st.expander("🚪 Exit model", expanded=False if compact_sidebar else True):
        exit_override = st.selectbox("Exit model", EXIT_MODELS, index=0, key="ui_exit_override")
        trail_atr_mult = st.number_input("Trailing ATR multiple", 0.5, 10.0, 2.0, 0.25, key="ui_trail_atr_mult")
        time_stop_bars = st.number_input("Time stop (bars)", 5, 5000, 48, key="ui_time_stop_bars")

    # ---- ML ----
    with st.expander("🤖 ML probability filter", expanded=False):
        use_ml = st.toggle("Enable ML filter (P(win) per signal)", False, key="ui_use_ml")
        ml_mode = st.selectbox("ML mode", ["Single-split (fast)", "Walk-forward (realistic)"], index=1, key="ui_ml_mode")
        train_frac = st.slider("Single-split training fraction", 0.3, 0.9, 0.7, key="ui_train_frac")
        wf_train_bars = st.number_input("Walk-forward train bars", 200, 20000, 3000, key="ui_wf_train_bars")
        wf_test_bars = st.number_input("Walk-forward test bars", 50, 5000, 500, key="ui_wf_test_bars")
        proba_thresh = st.slider("Only take trades if P(win) ≥", 0.50, 0.80, 0.55, 0.01, key="ui_proba_thresh")

    # ---- Chart options (performance matters) ----
    with st.expander("📈 Chart options (reduce lag)", expanded=False):
        chart_height = st.slider("Chart height", 520, 980, 720, 20, key="ui_chart_height")
        chart_window_mode = st.selectbox("Chart window", ["Last N candles (fast)", "Full backtest range (slower)"], index=0, key="ui_chart_window")
        chart_last_n = st.number_input("Last N candles", 300, 100000, 3500, step=100, key="ui_chart_last_n")
        chart_max_candles = st.number_input("Hard cap candles (safety)", 500, 200000, 12000, step=500, key="ui_chart_cap")

        chart_style = st.selectbox("Chart style", ["Candles", "Heikin Ashi", "OHLC", "Close line (fast)", "Close area (fast)"], index=0, key="ui_chart_style")

        show_indicators = st.toggle("Show indicator lines", True, key="ui_show_indicators")
        show_ml_overlay = st.toggle("Show ML P(win) markers", True, key="ui_show_ml_overlay")

        marker_mode = st.selectbox(
            "Trade markers",
            ["Selected trade only (fast)", "All trades (can lag)", "Off"],
            index=0,
            key="ui_marker_mode",
        )
        marker_limit = st.number_input("If 'All trades': cap markers to last N trades", 50, 20000, 2000, step=50, key="ui_marker_limit")
        show_selected_sl_tp = st.toggle("Show SL/TP for selected trade", True, key="ui_show_sl_tp")

        st.markdown("**Sessions (UTC overlays)**")
        show_sessions = st.toggle("Show session overlays", True, key="ui_show_sessions")
        show_asia = st.toggle("Asia (00–06)", True, key="ui_show_asia")
        show_london = st.toggle("London (07–10)", True, key="ui_show_london")
        show_ny = st.toggle("NY (13–16)", True, key="ui_show_ny")
        session_max_days = st.number_input("Auto-disable overlays if days >", 30, 3650, 180, step=30, key="ui_session_max_days")
        force_sessions = st.toggle("Force overlays even if slow", False, key="ui_force_sessions")

    st.divider()
    run_now = st.button("Run backtest", use_container_width=True)
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
    str(sizing_mode),
    float(start_balance),
    float(risk_pct),
    float(leverage_cap),
    float(min_pos_usd),
    float(max_pos_usd),
    bool(compound),
)


wins = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
losses = int((trades["pnl"] < 0).sum()) if not trades.empty else 0
net = float(trades["pnl"].sum()) if not trades.empty else 0.0
wr = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
avg_r = float(trades["r_mult"].mean()) if (not trades.empty and "r_mult" in trades.columns) else 0.0
avg_p = float(trades["p_win"].mean()) if (not trades.empty and "p_win" in trades.columns) else np.nan
avg_score = float(trades["score"].mean()) if (not trades.empty and "score" in trades.columns) else np.nan

st.markdown('<div class="muted">Tip: Use the sidebar sections to collapse settings. The chart tab also has performance controls (last N candles, markers, sessions).</div>', unsafe_allow_html=True)

tab_overview, tab_chart, tab_trades, tab_drill = st.tabs(["📌 Overview", "🕯️ Chart", "📋 Trades", "🔍 Drilldown"])

with tab_overview:
    st.subheader("Summary")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Trades", len(trades))
    c2.metric("Win rate", f"{wr:.1f}%")
    c3.metric("Wins", wins)
    c4.metric("Losses", losses)
    c5.metric("Net PnL ($)", f"{net:.0f}")
    c6.metric("Avg R", f"{avg_r:.2f}")
    c7.metric("Avg P(win)", f"{avg_p:.2f}" if np.isfinite(avg_p) else "—")
    c8.metric("Avg Score", f"{avg_score:.1f}" if np.isfinite(avg_score) else "—")


    if not trades.empty:
        colm1, colm2, colm3, colm4, colm5 = st.columns(5)
        with colm1:
            st.metric("Trades", len(trades))
        with colm2:
            st.metric("Win rate", f"{(trades['pnl']>0).mean()*100:.1f}%")
        with colm3:
            st.metric("Avg R", f"{trades['r_mult'].mean():.2f}")
        with colm4:
            st.metric("Net PnL ($)", f"{trades['pnl'].sum():,.2f}")
        with colm5:
            start_bal = float(st.session_state.get("ui_start_balance", 10_000.0))
            end_bal = float(trades["equity_after"].dropna().iloc[-1]) if ("equity_after" in trades.columns and trades["equity_after"].notna().any()) else float(start_bal + trades["pnl"].sum())
            st.metric("End balance", f"{end_bal:,.2f}")

        # Build equity series (uses equity_after if available)
        t = trades.sort_values("exit_time").copy()
        start_bal = float(st.session_state.get("ui_start_balance", 10_000.0))

        if "equity_after" in t.columns and t["equity_after"].notna().any():
            equity_series = pd.Series(t["equity_after"].astype(float).to_numpy(), index=t["exit_time"])
        else:
            equity_series = pd.Series((start_bal + t["pnl"].astype(float).cumsum()).to_numpy(), index=t["exit_time"])

        eq_df = pd.DataFrame({"time": equity_series.index, "equity": equity_series.values})
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["drawdown"] = eq_df["equity"] - eq_df["peak"]
        eq_df["drawdown_pct"] = np.where(eq_df["peak"] > 0, (eq_df["drawdown"] / eq_df["peak"]) * 100.0, 0.0)

        max_dd = float(eq_df["drawdown"].min()) if not eq_df.empty else 0.0
        max_dd_pct = float(eq_df["drawdown_pct"].min()) if not eq_df.empty else 0.0

        st.markdown("### Equity & drawdown")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scattergl(x=eq_df["time"], y=eq_df["equity"], mode="lines", name="Equity"))
        fig_eq.update_layout(height=280, template=plotly_template, margin=dict(l=10, r=10, t=40, b=10), title="Equity curve (balance)")
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scattergl(x=eq_df["time"], y=eq_df["drawdown"], mode="lines", name="Drawdown"))
        fig_dd.update_layout(height=220, template=plotly_template, margin=dict(l=10, r=10, t=40, b=10), title="Drawdown ($ from peak)")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.caption(f"Max drawdown: ${abs(max_dd):,.2f} ({max_dd_pct:.2f}%) from peak.")

        st.markdown("### Monthly performance")
        t["month"] = t["exit_time"].dt.to_period("M").dt.to_timestamp()
        mdf = (
            t.groupby("month")
            .agg(pnl=("pnl", "sum"), trades=("pnl", "size"), wins=("pnl", lambda s: int((s > 0).sum())))
            .reset_index()
        )
        mdf["win_rate"] = np.where(mdf["trades"] > 0, (mdf["wins"] / mdf["trades"]) * 100.0, np.nan)
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        # Compact year x month table (PnL)
        if not mdf.empty:
            tmp = mdf.copy()
            tmp["year"] = tmp["month"].dt.year
            tmp["mon"] = tmp["month"].dt.month
            pivot = tmp.pivot_table(index="year", columns="mon", values="pnl", aggfunc="sum").fillna(0.0)
            pivot.columns = [calendar.month_abbr[int(c)] for c in pivot.columns]
            st.dataframe(pivot, use_container_width=True)


with tab_chart:
    st.subheader("Chart")
    selected_abs = None
    if not trades.empty:
        t = trades.reset_index(drop=True)
        view_n = int(min(len(t), st.session_state.get("ui_marker_limit", 2000)))
        t_view = t.tail(view_n).reset_index(drop=True)

        labels = [
            f"#{(len(t)-len(t_view))+i+1} {r.side} | {r.entry_time:%Y-%m-%d %H:%M} → {r.exit_time:%Y-%m-%d %H:%M} | {r.outcome} | "
            f"Score {r.score:.0f} | P(win) {r.p_win:.2f} | R {r.r_mult:+.2f} | {r.sig_tag} | {r.exit_model}"
            for i, r in t_view.iterrows()
        ]
        selected = st.selectbox("Select a trade to highlight", ["(none)"] + labels, index=0)
        if selected != "(none)":
            rel = labels.index(selected)
            selected_abs = (len(t) - len(t_view)) + rel

    # ---- Chart windowing (performance) ----
    d_chart = d_full.copy()
    chart_window_mode = st.session_state.get("ui_chart_window", "Last N candles (fast)")
    last_n = int(st.session_state.get("ui_chart_last_n", 3500))
    cap_n = int(st.session_state.get("ui_chart_cap", 12000))

    d_plot = d_chart
    if str(chart_window_mode).startswith("Last"):
        d_plot = d_chart.iloc[-last_n:].copy()
    if len(d_plot) > cap_n:
        d_plot = d_plot.iloc[-cap_n:].copy()
        st.info(f"Chart capped to last {cap_n:,} candles for performance. Adjust in sidebar → Chart options.")

    # ---- Build figure ----
    fig = go.Figure()

    style = str(st.session_state.get("ui_chart_style", "Candles"))
    if style == "Close line (fast)":
        fig.add_trace(go.Scattergl(x=d_plot["dt"], y=d_plot["close"], mode="lines", name=f"BTCUSDT ({timeframe})"))
    elif style == "Close area (fast)":
        fig.add_trace(go.Scattergl(x=d_plot["dt"], y=d_plot["close"], mode="lines", fill="tozeroy", name=f"BTCUSDT ({timeframe})"))
    elif style == "OHLC":
        fig.add_trace(go.Ohlc(
            x=d_plot["dt"], open=d_plot["open"], high=d_plot["high"], low=d_plot["low"], close=d_plot["close"],
            name=f"BTCUSDT ({timeframe})"
        ))
    elif style == "Heikin Ashi":
        ha = heikin_ashi(d_plot)
        fig.add_trace(go.Candlestick(
            x=ha["dt"], open=ha["open"], high=ha["high"], low=ha["low"], close=ha["close"],
            name=f"BTCUSDT HA ({timeframe})"
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=d_plot["dt"], open=d_plot["open"], high=d_plot["high"], low=d_plot["low"], close=d_plot["close"],
            name=f"BTCUSDT ({timeframe})"
        ))

    # Sessions (can be expensive)
    show_sessions = bool(st.session_state.get("ui_show_sessions", True))
    force_sessions = bool(st.session_state.get("ui_force_sessions", False))
    session_max_days = int(st.session_state.get("ui_session_max_days", 180))
    if show_sessions:
        days = pd.to_datetime(d_plot["dt"]).dt.floor("D").nunique()
        if (days > session_max_days) and (not force_sessions):
            st.warning(f"Session overlays auto-disabled (days={days} > {session_max_days}). Enable 'Force overlays' in sidebar if needed.")
        else:
            fig = add_session_overlays(
                fig, d_plot["dt"],
                show_asia=bool(st.session_state.get("ui_show_asia", True)),
                show_london=bool(st.session_state.get("ui_show_london", True)),
                show_ny=bool(st.session_state.get("ui_show_ny", True)),
            )

    # Indicators (Scattergl for speed)
    if bool(st.session_state.get("ui_show_indicators", True)):
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
            if col_name in d_plot.columns:
                fig.add_trace(go.Scattergl(x=d_plot["dt"], y=d_plot[col_name], mode="lines", name=label))

    # Trade markers
    marker_mode = st.session_state.get("ui_marker_mode", "Selected trade only (fast)")
    show_sl_tp = bool(st.session_state.get("ui_show_sl_tp", True))
    marker_limit = int(st.session_state.get("ui_marker_limit", 2000))

    if not trades.empty and marker_mode != "Off":
        if str(marker_mode).startswith("All"):
            t_m = trades.tail(marker_limit).copy()
            fig.add_trace(go.Scattergl(
                x=t_m["entry_time"], y=t_m["entry_price"], mode="markers", name="Entries",
                marker=dict(symbol="triangle-up", size=8)
            ))
            fig.add_trace(go.Scattergl(
                x=t_m["exit_time"], y=t_m["exit_price"], mode="markers", name="Exits",
                marker=dict(symbol="x", size=8)
            ))
        elif selected_abs is not None:
            r = trades.reset_index(drop=True).iloc[int(selected_abs)]
            fig.add_trace(go.Scattergl(x=[r["entry_time"]], y=[r["entry_price"]], mode="markers", name="Entry",
                                      marker=dict(symbol="triangle-up", size=10)))
            fig.add_trace(go.Scattergl(x=[r["exit_time"]], y=[r["exit_price"]], mode="markers", name="Exit",
                                      marker=dict(symbol="x", size=10)))
            fig.add_trace(go.Scatter(
                x=[r["entry_time"], r["exit_time"]],
                y=[r["entry_price"], r["exit_price"]],
                mode="lines+markers",
                name="Selected trade",
                line=dict(width=4),
                marker=dict(size=9),
            ))
            if show_sl_tp:
                fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["sl"], r["sl"]], mode="lines", name="Selected SL"))
                fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["tp"], r["tp"]], mode="lines", name="Selected TP"))

    # ML overlay
    if bool(st.session_state.get("ui_show_ml_overlay", True)) and use_ml and (proba is not None) and ("long_sig" in d_plot.columns):
        sig_mask = (d_plot["long_sig"] | d_plot["short_sig"])
        sig_points = d_plot[sig_mask].copy()
        if len(sig_points) > 0:
            idx = sig_points.index.to_numpy()
            p = np.array(proba)[idx]
            fig.add_trace(go.Scattergl(
                x=sig_points["dt"],
                y=sig_points["close"],
                mode="markers",
                name="ML P(win) @ signals",
                marker=dict(
                    size=8, color=p, cmin=0.0, cmax=1.0, colorscale="Viridis",
                    colorbar=dict(title="P(win)"), symbol="circle", opacity=0.9
                ),
                text=[f"P(win)={pp:.2f}" if np.isfinite(pp) else "P(win)=NA" for pp in p],
                hovertemplate="%{text}<br>%{x}<br>Close=%{y}<extra></extra>",
            ))

    fig.update_layout(
        height=int(st.session_state.get("ui_chart_height", 720)),
        template=plotly_template,
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False})

with tab_trades:
    st.subheader("Trades Table")
    if trades.empty:
        st.write("No trades in this range with current settings.")
    else:
        tshow = trades.copy()

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            side_filter = st.multiselect("Side", ["LONG", "SHORT"], default=["LONG", "SHORT"])
        with fcol2:
            outcomes = sorted(tshow["outcome"].dropna().unique().tolist())
            outcome_filter = st.multiselect("Outcome", outcomes, default=outcomes)
        with fcol3:
            min_score_f = st.slider("Min score (filter)", 0.0, 100.0, 0.0, 1.0)
        with fcol4:
            min_r_f = st.slider("Min R (filter)", -5.0, 10.0, -5.0, 0.25)

        tshow = tshow[tshow["side"].isin(side_filter)]
        tshow = tshow[tshow["outcome"].isin(outcome_filter)]
        if "score" in tshow.columns:
            tshow = tshow[tshow["score"].fillna(0) >= float(min_score_f)]
        if "r_mult" in tshow.columns:
            tshow = tshow[tshow["r_mult"].fillna(-999) >= float(min_r_f)]

        for col in ["signal_time", "entry_time", "exit_time"]:
            tshow[col] = pd.to_datetime(tshow[col]).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(tshow, use_container_width=True, hide_index=True)

        st.download_button(
            "Download trades CSV",
            data=trades.to_csv(index=False).encode("utf-8"),
            file_name=f"trades_{timeframe}_{strategy_name.replace(' ', '_')}.csv",
            mime="text/csv",
        )

with tab_drill:
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

            ps = reason.get("position_sizing", {})
            if ps:
                st.write("**Position sizing at entry:**")
                st.write(f"- Mode: {ps.get('mode','')}")
                st.write(f"- Equity ref: {ps.get('equity_ref','—')}")
                st.write(f"- Risk target: {ps.get('risk_target','—')}")
                st.write(f"- Risk actual: {ps.get('risk_actual','—')}")
                st.write(f"- Qty (BTC): {ps.get('qty','—')}")
                st.write(f"- Notional ($): {ps.get('notional','—')}")
                st.write(f"- Leverage used: {ps.get('leverage_used','—')} (cap {ps.get('leverage_cap','—')})")
                st.write(f"- Capped: {ps.get('size_capped', False)} | Min-forced: {ps.get('min_forced', False)}")

            ind = reason.get("indicators", {})
            st.write("**Indicators at signal candle:**")
            st.write(
                f"- RSI: {ind.get('RSI','—')}\n"
                f"- ROC: {ind.get('ROC','—')}\n"
                f"- VOL expansion: {ind.get('VOL_EXP','—')}\n"
                f"- ATR: {ind.get('ATR','—')}"
            )

        with col2:
            st.markdown("### Exit & outcome")
            st.write(f"**Exit model:** {tr.get('exit_model','')}")
            st.write(f"**Outcome:** {tr.get('outcome','')}")
            st.write(f"**R multiple:** {tr.get('r_mult','—'):.2f}" if pd.notna(tr.get("r_mult", np.nan)) else "—")
            st.write(f"**PnL ($):** {tr.get('pnl','—')}")
            st.write(f"**Bars held:** {int(tr.get('bars_in_trade',0))}")
            if "mfe" in tr and "mae" in tr:
                st.write(f"**MFE:** {tr.get('mfe','—'):.2f}  |  **MAE:** {tr.get('mae','—'):.2f}")

            st.markdown("**Why it won/lost (model-based):**")
            if str(tr.get("outcome","")) in ["TP"]:
                st.success("Price reached your take-profit before your stop (according to the candle OHLC model).")
            elif "SL" in str(tr.get("outcome","")):
                st.error("Price hit your stop-loss (or trailing stop) before take-profit (according to candle OHLC).")
            elif str(tr.get("outcome","")) == "EMA_FLIP":
                st.warning("The EMA trend flipped against the trade, so the exit model closed it early.")
            elif str(tr.get("outcome","")) == "TIME":
                st.warning("The time-stop closed the trade after the max holding period.")
            else:
                st.info("The position closed because an opposite signal/exit condition triggered.")

        st.markdown("### Confluence score breakdown")
        comps = reason.get("components", {})
        wts = reason.get("weights", {})
        contrib = reason.get("contribution", {})

        if comps:
            rows = []
            for k, v in comps.items():
                rows.append({
                    "component": k,
                    "value(0-1)": float(v) if v is not None else np.nan,
                    "weight": float(wts.get(k, 0.0)),
                    "weighted_contrib": float(contrib.get(k, 0.0)),
                })
            df_break = pd.DataFrame(rows).sort_values("weighted_contrib", ascending=False)
            st.dataframe(df_break, use_container_width=True, hide_index=True)
        else:
            st.write("No component breakdown available for this trade (check `score_components`).")
