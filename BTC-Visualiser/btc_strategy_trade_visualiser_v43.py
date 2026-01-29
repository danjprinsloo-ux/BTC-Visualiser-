# BTC Strategy Trade Visualiser (v4.6)
# - Realistic entry: signal on candle close -> fill next candle open (default ON)
# - Intrabar TP/SL tie-break: worst/best
# - NEW: ML probability scoring filter (logistic regression; no sklearn)
# Run with: python -m streamlit run btc_strategy_trade_visualiser_v43.py

import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BTC Strategy Trade Visualiser (v4.6)", layout="wide")


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


# =====================
# ML: Logistic Regression (no sklearn)
# =====================
def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def fit_logreg(X, y, lr=0.2, steps=800, l2=1e-2, seed=7):
    """
    Simple logistic regression with L2 regularization.
    X: (n, d), y: (n,)
    Returns (w, b, mu, sd)
    """
    rng = np.random.default_rng(seed)

    # standardize
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd

    # drop NaN rows
    mask = np.isfinite(Xs).all(axis=1) & np.isfinite(y)
    Xs = Xs[mask]
    y = y[mask].astype(float)

    if len(y) < 200:
        return None

    w = rng.normal(0, 0.01, size=Xs.shape[1])
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(Xs @ w + b)
        # gradients
        err = p - y
        gw = (Xs.T @ err) / len(y) + l2 * w
        gb = np.mean(err)
        # update
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


# =====================
# FEATURES for ML (available at signal candle close)
# =====================
def build_ml_features(df: pd.DataFrame, fast: int, slow: int, trend_len: int, atr_len: int):
    """
    Features computed at each candle using only historical info up to that close.
    """
    d = df.copy()
    d["FAST"] = sma(d["close"], fast)
    d["SLOW"] = sma(d["close"], slow)
    d["TREND"] = sma(d["close"], trend_len)
    d["ATR"] = atr(d, atr_len)

    ret = d["close"].pct_change()
    d["ret_std"] = ret.rolling(50).std()

    # slope proxy (trend_len MA change over last 10 bars)
    slope_len = 10
    d["trend_slope"] = (d["TREND"] - d["TREND"].shift(slope_len)) / slope_len

    # normalized features
    d["atr_pct"] = d["ATR"] / d["close"]
    d["slope_pct"] = d["trend_slope"] / d["close"]
    d["spread_pct"] = (d["FAST"] - d["SLOW"]) / d["close"]
    d["dist_trend_pct"] = (d["close"] - d["TREND"]) / d["close"]

    cols = ["atr_pct", "ret_std", "slope_pct", "spread_pct", "dist_trend_pct"]
    X = d[cols].to_numpy(dtype=float)
    return X, cols, d


# =====================
# BACKTEST (with optional ML probability filter)
# =====================
def backtest(
    df, fast, slow, trend_len, atr_len, atr_mult, rr, risk, both_dirs, trend_filter, exit_on_opp,
    fill_next_open=True, intrabar_rule="Worst case (SL first)",
    # ML filter
    use_ml=False, proba=None, proba_thresh=0.55
):
    d = df.copy()

    d["FAST"] = sma(d["close"], fast)
    d["SLOW"] = sma(d["close"], slow)
    d["TREND"] = sma(d["close"], trend_len) if trend_filter else np.nan
    d["ATR"] = atr(d, atr_len)

    trades = []
    pos = 0
    entry = sl = tp = None
    entry_time = None
    signal_time = None
    signal_proba = None

    start_i = max(fast, slow, atr_len, (trend_len if trend_filter else 0)) + 2
    end_limit = len(d) - 2 if fill_next_open else len(d) - 1

    if len(d) <= start_i + 2:
        return d, pd.DataFrame()

    intrabar_is_best = intrabar_rule.lower().startswith("best")

    for i in range(start_i, end_limit + 1):
        row = d.iloc[i]
        prev = d.iloc[i - 1]

        if pd.isna(row["ATR"]) or pd.isna(prev["FAST"]) or pd.isna(prev["SLOW"]) or pd.isna(row["FAST"]) or pd.isna(row["SLOW"]):
            continue

        cross_up = prev["FAST"] <= prev["SLOW"] and row["FAST"] > row["SLOW"]
        cross_dn = prev["FAST"] >= prev["SLOW"] and row["FAST"] < row["SLOW"]

        long_sig = cross_up
        short_sig = cross_dn

        if trend_filter and pd.notna(row["TREND"]):
            long_sig = long_sig and (row["close"] > row["TREND"])
            short_sig = short_sig and (row["close"] < row["TREND"])

        # ML gating at signal candle i
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

        # Determine fill
        if fill_next_open:
            fill_row = d.iloc[i + 1]
            fill_price = float(fill_row["open"])
            fill_time = fill_row["dt"]
        else:
            fill_price = float(row["close"])
            fill_time = row["dt"]

        if pos == 0:
            if long_sig and allowed():
                pos = 1
                entry = float(fill_price)
                entry_time = fill_time
                signal_time = row["dt"]
                signal_proba = pwin
                sl = entry - atr_mult * float(row["ATR"])
                tp = entry + rr * (entry - sl)

            elif both_dirs and short_sig and allowed():
                pos = -1
                entry = float(fill_price)
                entry_time = fill_time
                signal_time = row["dt"]
                signal_proba = pwin
                sl = entry + atr_mult * float(row["ATR"])
                tp = entry - rr * (sl - entry)

        else:
            if fill_next_open and row["dt"] < entry_time:
                continue

            if pos == 1:
                stop_hit = row["low"] <= sl
                tp_hit = row["high"] >= tp
                opp_hit = exit_on_opp and short_sig
            else:
                stop_hit = row["high"] >= sl
                tp_hit = row["low"] <= tp
                opp_hit = exit_on_opp and long_sig

            if stop_hit or tp_hit or opp_hit:
                if stop_hit and tp_hit:
                    if intrabar_is_best:
                        stop_hit = False
                    else:
                        tp_hit = False

                if stop_hit:
                    exit_price = float(sl)
                    pnl = -float(risk)
                    outcome = "SL"
                elif tp_hit:
                    exit_price = float(tp)
                    pnl = float(risk) * float(rr)
                    outcome = "TP"
                else:
                    exit_price = float(row["close"])
                    if pos == 1:
                        r = entry - sl
                        pnl = float(risk) * ((exit_price - entry) / r) if r > 0 else 0.0
                    else:
                        r = sl - entry
                        pnl = float(risk) * ((entry - exit_price) / r) if r > 0 else 0.0
                    outcome = "Opp"

                if pos == 1:
                    r_dist = entry - sl
                    r_mult = (exit_price - entry) / r_dist if r_dist > 0 else 0.0
                else:
                    r_dist = sl - entry
                    r_mult = (entry - exit_price) / r_dist if r_dist > 0 else 0.0

                trades.append(
                    {
                        "side": "LONG" if pos == 1 else "SHORT",
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
                    }
                )

                pos = 0
                entry = sl = tp = None
                entry_time = None
                signal_time = None
                signal_proba = None

    return d, pd.DataFrame(trades)


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    df_view, fast, slow, trend_len, atr_len, atr_mult, rr, risk,
    both_dirs, trend_filter, exit_on_opp,
    fill_next_open, intrabar_rule,
    use_ml, proba, proba_thresh
):
    return backtest(
        df_view, fast, slow, trend_len, atr_len, atr_mult, rr, risk,
        both_dirs, trend_filter, exit_on_opp,
        fill_next_open=fill_next_open, intrabar_rule=intrabar_rule,
        use_ml=use_ml, proba=proba, proba_thresh=proba_thresh
    )


# =====================
# UI
# =====================
st.title("BTC Strategy Trade Visualiser (v4.6) — ML probability filter")

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
    st.header("Timeframe")
    tf_label_to_rule = {"1m": "1T", "5m": "5T", "10m": "10T", "15m": "15T", "1h": "1H", "4h": "4H"}
    timeframe = st.selectbox("Chart/Backtest timeframe", list(tf_label_to_rule.keys()), index=3)

    st.divider()
    st.header("Strategy")
    fast = st.number_input("Fast SMA", 2, 500, 10)
    slow = st.number_input("Slow SMA", 3, 1000, 20)
    trend_len = st.number_input("Trend SMA", 10, 2000, 200)
    atr_len = st.number_input("ATR Length", 2, 100, 14)
    atr_mult = st.number_input("ATR Multiplier", 0.5, 10.0, 2.5)
    rr = st.number_input("Risk Reward (RR)", 0.5, 10.0, 2.0)
    risk = st.number_input("Risk per trade ($)", 1.0, 1_000_000.0, 100.0)

    both_dirs = st.toggle("Trade Long & Short", True)
    trend_filter = st.toggle("Use Trend Filter", True)
    exit_on_opp = st.toggle("Exit on opposite signal", False)

    st.divider()
    st.header("Execution realism")
    fill_next_open = st.toggle("Enter on next candle OPEN (more realistic)", True)
    intrabar_rule = st.selectbox("If TP & SL hit the same candle", ["Worst case (SL first)", "Best case (TP first)"], index=0)

    st.divider()
    st.header("ML Probability Filter")
    use_ml = st.toggle("Enable ML filter (score each signal)", False)
    train_frac = st.slider("Training fraction (first part of range)", 0.3, 0.9, 0.7)
    proba_thresh = st.slider("Only take trades if P(win) ≥", 0.50, 0.80, 0.55, 0.01)
    st.caption("ML is trained on past signals only, then used to filter future signals in the same run.")

    st.divider()
    st.header("Chart")
    show_smas = st.toggle("Show SMA lines", True)
    show_selected_sl_tp = st.toggle("Show SL/TP for selected trade", True)

    st.divider()
    st.header("Performance")
    run_now = st.button("Run backtest")


if not files:
    st.info("Upload Binance BTCUSDT files to begin.")
    st.stop()

file_payload = [(f.name, f.getvalue()) for f in files]
df = load_files_cached(file_payload)

if df.empty:
    st.error("No valid Binance data loaded.")
    st.stop()

base_min = infer_base_minutes(df)
st.caption(f"Loaded rows: {len(df):,} | Data range: {df['dt'].min()} → {df['dt'].max()} | Base ~ {base_min:.2f} min" if base_min else "")

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


# =====================
# ML training + probability array (optional)
# =====================
proba = None
ml_info = None

if use_ml:
    # Build features for entire df_tf
    X_all, feat_cols, d_feat = build_ml_features(df_tf, int(fast), int(slow), int(trend_len), int(atr_len))

    # Determine training split
    cut_idx = int(len(df_tf) * float(train_frac))
    cut_idx = max(50, min(cut_idx, len(df_tf) - 50))
    train_end_dt = df_tf["dt"].iloc[cut_idx]

    # First: generate TRAINING trades WITHOUT ML filter (so we have labels)
    d_train = df_tf[df_tf["dt"] <= train_end_dt].copy()
    _, trades_train = backtest(
        d_train, int(fast), int(slow), int(trend_len), int(atr_len), float(atr_mult), float(rr), float(risk),
        bool(both_dirs), bool(trend_filter), bool(exit_on_opp),
        fill_next_open=bool(fill_next_open), intrabar_rule=str(intrabar_rule),
        use_ml=False
    )

    if trades_train.empty:
        st.warning("ML: No trades in training segment; cannot train ML. Running without ML filter.")
        use_ml = False
    else:
        # Labels: 1 if profitable (R > 0), 0 if not
        # Exclude 'Opp' exits to reduce noise (optional)
        tt = trades_train.copy()
        tt = tt[tt["outcome"].isin(["TP", "SL"])].copy()
        if tt.empty:
            st.warning("ML: Training trades are all Opp exits; cannot train reliably. Running without ML filter.")
            use_ml = False
        else:
            y = (tt["r_mult"].to_numpy(dtype=float) > 0).astype(float)

            # Map each training trade to feature row at its signal_time
            dt_to_idx = {t: i for i, t in enumerate(d_train["dt"].tolist())}
            idxs = []
            for stime in tt["signal_time"].tolist():
                idx = dt_to_idx.get(stime, None)
                if idx is None:
                    idx = int(np.searchsorted(d_train["dt"].values, stime))
                    idx = max(0, min(idx, len(d_train) - 1))
                idxs.append(idx)

            X = X_all[:len(d_train)][np.array(idxs, dtype=int)]

            model = fit_logreg(X, y, lr=0.2, steps=800, l2=1e-2, seed=7)
            if model is None:
                st.warning("ML: Not enough clean training rows for ML. Running without ML filter.")
                use_ml = False
            else:
                proba = predict_logreg_proba(X_all, model)
                ml_info = f"ML trained on signals ≤ {pd.Timestamp(train_end_dt):%Y-%m-%d}. Features: {', '.join(feat_cols)}"

if ml_info:
    st.info(ml_info)


# =====================
# Backtest (with optional ML proba filter)
# =====================
d_full, trades = run_backtest_cached(
    df_tf, int(fast), int(slow), int(trend_len), int(atr_len), float(atr_mult), float(rr), float(risk),
    bool(both_dirs), bool(trend_filter), bool(exit_on_opp),
    bool(fill_next_open), str(intrabar_rule),
    bool(use_ml), proba, float(proba_thresh)
)

wins = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
losses = int((trades["pnl"] < 0).sum()) if not trades.empty else 0
net = float(trades["pnl"].sum()) if not trades.empty else 0.0
wr = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
avg_r = float(trades["r_mult"].mean()) if (not trades.empty and "r_mult" in trades.columns) else 0.0
avg_p = float(trades["p_win"].mean()) if (not trades.empty and "p_win" in trades.columns) else np.nan

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Trades", len(trades))
c2.metric("Win rate", f"{wr:.1f}%")
c3.metric("Wins", wins)
c4.metric("Losses", losses)
c5.metric("Net PnL ($)", f"{net:.0f}")
c6.metric("Avg R", f"{avg_r:.2f}")
c7.metric("Avg P(win)", f"{avg_p:.2f}" if np.isfinite(avg_p) else "—")

selected_idx = None
t = None
if not trades.empty:
    t = trades.reset_index(drop=True)
    labels = [
        f"#{i+1} {r.side} | {r.entry_time:%Y-%m-%d %H:%M} → {r.exit_time:%Y-%m-%d %H:%M} | {r.outcome} | P(win) {r.p_win:.2f} | R {r.r_mult:+.2f}"
        for i, r in t.iterrows()
    ]
    selected = st.selectbox("Select a trade to highlight", ["(none)"] + labels, index=0)
    if selected != "(none)":
        selected_idx = labels.index(selected)

# ===== chart =====
d_chart = d_full.copy()

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=d_chart["dt"], open=d_chart["open"], high=d_chart["high"], low=d_chart["low"], close=d_chart["close"],
    name=f"BTCUSDT ({timeframe})"
))

if show_smas:
    fig.add_trace(go.Scatter(x=d_chart["dt"], y=d_chart["FAST"], mode="lines", name=f"SMA {fast}"))
    fig.add_trace(go.Scatter(x=d_chart["dt"], y=d_chart["SLOW"], mode="lines", name=f"SMA {slow}"))
    if trend_filter:
        fig.add_trace(go.Scatter(x=d_chart["dt"], y=d_chart["TREND"], mode="lines", name=f"SMA {trend_len}"))

if t is not None and not t.empty:
    fig.add_trace(go.Scatter(
        x=t["entry_time"], y=t["entry_price"],
        mode="markers", name="Entries", marker=dict(symbol="triangle-up", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=t["exit_time"], y=t["exit_price"],
        mode="markers", name="Exits", marker=dict(symbol="x", size=10),
    ))

if selected_idx is not None and t is not None:
    r = t.iloc[selected_idx]
    fig.add_trace(go.Scatter(
        x=[r["entry_time"], r["exit_time"]],
        y=[r["entry_price"], r["exit_price"]],
        mode="lines+markers",
        name="Selected trade",
        line=dict(width=4),
        marker=dict(size=11),
    ))
    if show_selected_sl_tp:
        fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["sl"], r["sl"]], mode="lines", name="Selected SL"))
        fig.add_trace(go.Scatter(x=[r["entry_time"], r["exit_time"]], y=[r["tp"], r["tp"]], mode="lines", name="Selected TP"))

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
        file_name=f"trades_{timeframe}.csv",
        mime="text/csv",
    )
