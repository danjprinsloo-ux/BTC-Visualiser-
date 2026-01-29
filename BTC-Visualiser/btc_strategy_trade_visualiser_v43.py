# BTC Strategy Trade Visualiser (v4.2)
# - Crosshair always ON + Pan default (no toggling)
# - Timeframe selector with safe resampling (needs base timeframe <= target)
# Run with: python -m streamlit run btc_strategy_trade_visualiser_v42.py

import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BTC Strategy Trade Visualiser (v4.2)", layout="wide")


# =====================
# LOADER
# =====================
def _read_binance_like_csv(fileobj):
    """
    Binance kline CSVs commonly come as:
    open_time, open, high, low, close, volume, close_time, ...
    We keep first 6 columns.
    """
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

    # numeric coercion
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
    """
    rule: pandas offset alias (e.g. '15T', '1H', '4H', '5T', '1T')
    """
    d = df.set_index("dt").sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
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
# BACKTEST
# =====================
def backtest(df, fast, slow, trend_len, atr_len, atr_mult, rr, risk, both_dirs, trend_filter, exit_on_opp):
    d = df.copy()

    d["FAST"] = sma(d["close"], fast)
    d["SLOW"] = sma(d["close"], slow)
    d["TREND"] = sma(d["close"], trend_len) if trend_filter else np.nan
    d["ATR"] = atr(d, atr_len)

    trades = []
    pos = 0
    entry = sl = tp = None
    entry_time = None

    start_i = max(fast, slow, atr_len, (trend_len if trend_filter else 0)) + 2
    if len(d) <= start_i:
        return d, pd.DataFrame()

    for i in range(start_i, len(d)):
        row = d.iloc[i]
        prev = d.iloc[i - 1]

        if pd.isna(row["ATR"]) or pd.isna(prev["FAST"]) or pd.isna(prev["SLOW"]):
            continue

        cross_up = prev["FAST"] <= prev["SLOW"] and row["FAST"] > row["SLOW"]
        cross_dn = prev["FAST"] >= prev["SLOW"] and row["FAST"] < row["SLOW"]

        long_sig = cross_up
        short_sig = cross_dn

        if trend_filter and pd.notna(row["TREND"]):
            long_sig = long_sig and (row["close"] > row["TREND"])
            short_sig = short_sig and (row["close"] < row["TREND"])

        if pos == 0:
            if long_sig:
                pos = 1
                entry = float(row["close"])
                entry_time = row["dt"]
                sl = entry - atr_mult * float(row["ATR"])
                tp = entry + rr * (entry - sl)

            elif both_dirs and short_sig:
                pos = -1
                entry = float(row["close"])
                entry_time = row["dt"]
                sl = entry + atr_mult * float(row["ATR"])
                tp = entry - rr * (sl - entry)

        else:
            if pos == 1:
                stop_hit = row["low"] <= sl
                tp_hit = row["high"] >= tp
                opp_hit = exit_on_opp and short_sig
            else:
                stop_hit = row["high"] >= sl
                tp_hit = row["low"] <= tp
                opp_hit = exit_on_opp and long_sig

            if stop_hit or tp_hit or opp_hit:
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

                trades.append(
                    {
                        "side": "LONG" if pos == 1 else "SHORT",
                        "entry_time": entry_time,
                        "exit_time": row["dt"],
                        "entry_price": float(entry),
                        "exit_price": float(exit_price),
                        "sl": float(sl),
                        "tp": float(tp),
                        "rr": float(rr),
                        "risk": float(risk),
                        "pnl": float(pnl),
                        "outcome": outcome,
                    }
                )

                pos = 0
                entry = sl = tp = None
                entry_time = None

    return d, pd.DataFrame(trades)


@st.cache_data(show_spinner=False)
def run_backtest_cached(df_view, fast, slow, trend_len, atr_len, atr_mult, rr, risk, both_dirs, trend_filter, exit_on_opp):
    return backtest(df_view, fast, slow, trend_len, atr_len, atr_mult, rr, risk, both_dirs, trend_filter, exit_on_opp)


# =====================
# UI
# =====================
st.title("BTC Strategy Trade Visualiser (v4.2) — full candles + timeframe")

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
    tf_label_to_rule = {
        "1m": "1T",
        "5m": "5T",
        "10m": "10T",
        "15m": "15T",
        "1h": "1H",
        "4h": "4H",
    }
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
    st.header("Chart")
    show_smas = st.toggle("Show SMA lines", True)
    show_selected_sl_tp = st.toggle("Show SL/TP for selected trade", True)

    st.caption("Crosshair is always ON. Drag mode defaults to PAN (like an exchange).")

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

date_range = st.slider(
    "Date range (UTC)",
    min_value=min_dt,
    max_value=max_dt,
    value=(min_dt, max_dt),
)

df_view_full = df[(df["dt"] >= date_range[0]) & (df["dt"] <= date_range[1])].copy()

# timeframe feasibility
target_rule = tf_label_to_rule[timeframe]
target_minutes = {
    "1T": 1, "5T": 5, "10T": 10, "15T": 15, "1H": 60, "4H": 240
}[target_rule]

if base_min is None:
    st.error("Could not infer the base timeframe from your data.")
    st.stop()

# You can only resample to >= base timeframe
if target_minutes < (base_min - 0.01):
    st.error(
        f"Your uploaded data is ~{base_min:.0f}m candles. "
        f"You cannot create {timeframe} from that. "
        f"Download and upload {timeframe} (or smaller, e.g. 1m) data."
    )
    st.stop()

# resample if needed
if abs(base_min - target_minutes) < 0.5:
    df_tf = df_view_full.copy()
else:
    df_tf = resample_cached(df_view_full, target_rule)

if df_tf.empty:
    st.error("No candles after timeframe conversion in this date range.")
    st.stop()

# run
if "has_run" not in st.session_state:
    st.session_state.has_run = False

if not run_now and st.session_state.has_run is False:
    run_now = True

if not run_now:
    st.info("Adjust settings, then click **Run backtest**.")
    st.stop()

st.session_state.has_run = True

d_full, trades = run_backtest_cached(
    df_tf, fast, slow, trend_len, atr_len, atr_mult, rr, risk, both_dirs, trend_filter, exit_on_opp
)

wins = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
losses = int((trades["pnl"] < 0).sum()) if not trades.empty else 0
net = float(trades["pnl"].sum()) if not trades.empty else 0.0
wr = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades", len(trades))
c2.metric("Win rate", f"{wr:.1f}%")
c3.metric("Wins", wins)
c4.metric("Losses", losses)
c5.metric("Net PnL ($)", f"{net:.0f}")

selected_idx = None
t = None
if not trades.empty:
    t = trades.reset_index(drop=True)
    labels = [
        f"#{i+1} {r.side} | {r.entry_time:%Y-%m-%d %H:%M} → {r.exit_time:%Y-%m-%d %H:%M} | {r.outcome} | PnL {r.pnl:+.0f}"
        for i, r in t.iterrows()
    ]
    selected = st.selectbox("Select a trade to highlight", ["(none)"] + labels, index=0)
    if selected != "(none)":
        selected_idx = labels.index(selected)

# ===== chart (ALL candles, NO downsample) =====
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
        mode="markers", name="Entries",
        marker=dict(symbol="triangle-up", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=t["exit_time"], y=t["exit_price"],
        mode="markers", name="Exits",
        marker=dict(symbol="x", size=10),
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
        fig.add_trace(go.Scatter(
            x=[r["entry_time"], r["exit_time"]],
            y=[r["sl"], r["sl"]],
            mode="lines",
            name="Selected SL",
        ))
        fig.add_trace(go.Scatter(
            x=[r["entry_time"], r["exit_time"]],
            y=[r["tp"], r["tp"]],
            mode="lines",
            name="Selected TP",
        ))

# Crosshair + pan together:
# - dragmode = 'pan'
# - hovermode x unified + spikes = crosshair feel
fig.update_layout(
    height=720,
    dragmode="pan",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
)

fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False},
)

st.subheader("Trades Table")
if trades.empty:
    st.write("No trades in this range with current settings.")
else:
    tshow = trades.copy()
    tshow["entry_time"] = tshow["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    tshow["exit_time"] = tshow["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(tshow, use_container_width=True, hide_index=True)

    st.download_button(
        "Download trades CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{timeframe}.csv",
        mime="text/csv",
    )
