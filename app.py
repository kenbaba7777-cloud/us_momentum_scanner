
import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="US Momentum Scanner", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.stApp { background: #07101d; color: #eef4ff; }
.block-container { max-width: 1500px; padding-top: 1.5rem; }
h1 { font-size: 2.5rem !important; }
.metric-card {
    background: #0d1929; border: 1px solid #1d3148; border-radius: 14px;
    padding: 14px; min-height: 95px;
}
.small { color:#91a5bd; font-size:.85rem; }
.score { font-size:1.7rem; font-weight:800; }
.good { color:#39e58c; } .bad { color:#ff5d6c; } .warn { color:#ffd166; }
table { font-size: 14px; }
@media (max-width: 700px) {
  h1 { font-size: 1.8rem !important; }
  .block-container { padding: .7rem; }
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 US Momentum & Breakout Scanner")
st.caption("Technische Rangliste für kurzfristige Momentum-Setups – kein Kursziel und keine Garantie.")

# ---------------- Universe ----------------
@st.cache_data(ttl=86400)
def sp500():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return t["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","TSLA","JPM","LLY"]

@st.cache_data(ttl=86400)
def nasdaq100():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        col = next(c for c in t.columns if "Ticker" in str(c))
        return t[col].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception:
        return ["NVDA","AMD","AVGO","MU","PLTR","APP","MSTR","NFLX","COST","QCOM"]

@st.cache_data(ttl=86400)
def russell2000():
    # Best effort: current IWM holdings CSV, with fallback.
    url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        text = r.text.replace("\x00", "")
        df = pd.read_csv(io.StringIO(text), skiprows=9)
        if "Ticker" in df.columns:
            return df["Ticker"].dropna().astype(str).str.strip().tolist()
    except Exception:
        pass
    return ["RKLB","IONQ","SOUN","HIMS","CELH","ASTS","CRDO","TMDX","SMCI","CAVA","RXRX","ENVX"]

def build_universe(choice):
    s = set(sp500())
    if "Nasdaq" in choice:
        s.update(nasdaq100())
    if "Russell" in choice:
        s.update(russell2000())
    s.update(["SPY", "QQQ"])
    return sorted(x for x in s if x and x.upper() not in {"NAN","USD"})

# ---------------- Data ----------------
@st.cache_data(ttl=300, show_spinner=False)
def download_prices(tickers):
    data = yf.download(
        tickers=tickers, period="1y", interval="1d",
        auto_adjust=True, progress=False, threads=True
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
        volume = data["Volume"].copy()
    else:
        close = data[["Close"]].copy()
        volume = data[["Volume"]].copy()
    close = close.dropna(axis=1, how="all")
    volume = volume.reindex(columns=close.columns)
    return close, volume

def calc_scanner(close, volume, min_price=5.0, min_dollar_vol=2_000_000):
    rows = []
    spy = close["SPY"].dropna() if "SPY" in close else pd.Series(dtype=float)
    if spy.empty:
        return pd.DataFrame()

    spy_5 = spy.pct_change(5).iloc[-1]
    spy_20 = spy.pct_change(20).iloc[-1]

    for ticker in close.columns:
        if ticker in ("SPY","QQQ"):
            continue
        s = close[ticker].dropna()
        v = volume[ticker].reindex(s.index).fillna(0) if ticker in volume else pd.Series(0, index=s.index)
        if len(s) < 65:
            continue

        price = float(s.iloc[-1])
        avg_vol20 = float(v.tail(20).mean())
        dollar_vol = float((s.tail(20) * v.tail(20)).mean())
        if price < min_price or dollar_vol < min_dollar_vol or avg_vol20 <= 0:
            continue

        r1 = s.pct_change(1).iloc[-1]
        r3 = s.pct_change(3).iloc[-1]
        r5 = s.pct_change(5).iloc[-1]
        r20 = s.pct_change(20).iloc[-1]
        r60 = s.pct_change(60).iloc[-1]

        ma20 = s.rolling(20).mean().iloc[-1]
        ma50 = s.rolling(50).mean().iloc[-1]
        ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan

        vol_ratio = float(v.iloc[-1] / avg_vol20) if avg_vol20 else 0
        high20 = s.tail(20).max()
        high52 = s.tail(252).max()
        dist52 = price / high52 - 1
        breakout20 = price >= high20 * 0.995
        near52 = price >= high52 * 0.97

        atr = s.diff().abs().rolling(14).mean().iloc[-1]
        atr_pct = float(atr / price) if price else 0

        # Acceleration: short-term momentum stronger than medium-term pace.
        accel = r5 - (r20 / 4.0)

        rs5 = r5 - spy_5
        rs20 = r20 - spy_20

        trend = (
            (price > ma20) * 25 +
            (price > ma50) * 25 +
            ((price > ma200) if not np.isnan(ma200) else 0) * 25 +
            (ma20 > ma50) * 25
        )

        momentum = np.clip(
            0.35 * ((r5 + 0.10) / 0.20) +
            0.40 * ((r20 + 0.20) / 0.40) +
            0.25 * ((r60 + 0.30) / 0.60), 0, 1
        ) * 100

        rs_score = np.clip(
            0.55 * ((rs5 + 0.08) / 0.16) +
            0.45 * ((rs20 + 0.15) / 0.30), 0, 1
        ) * 100

        volume_score = np.clip((vol_ratio - 0.8) / 3.2, 0, 1) * 100
        breakout_score = (70 if breakout20 else 25) + (30 if near52 else 0)
        acceleration_score = np.clip((accel + 0.04) / 0.12, 0, 1) * 100

        score = (
            0.25 * momentum +
            0.20 * rs_score +
            0.20 * volume_score +
            0.15 * breakout_score +
            0.10 * trend +
            0.10 * acceleration_score
        )

        setup = []
        if breakout20: setup.append("20D Breakout")
        if near52: setup.append("52W High")
        if vol_ratio >= 2: setup.append("Volume Surge")
        if accel > 0.03: setup.append("Acceleration")
        if price > ma20 > ma50: setup.append("Trend")
        if rs5 > 0.03: setup.append("RS")

        rows.append({
            "Ticker": ticker, "Score": round(score,1), "Price": price,
            "1D %": r1*100, "3D %": r3*100, "5D %": r5*100,
            "20D %": r20*100, "60D %": r60*100,
            "Vol x": vol_ratio, "RS 5D %": rs5*100, "RS 20D %": rs20*100,
            "52W High %": dist52*100, "Dollar Vol $": dollar_vol,
            "ATR %": atr_pct*100, "Setup": " • ".join(setup) or "Momentum"
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Scanner")
    universe_choice = st.selectbox(
        "Universum",
        ["S&P 500 + Nasdaq-100 + Russell 2000", "S&P 500 + Nasdaq-100", "S&P 500", "Russell 2000"]
    )
    min_price = st.number_input("Mindestpreis ($)", 0.0, 500.0, 5.0, 1.0)
    min_dollar = st.number_input("Min. Ø Tagesvolumen ($)", 0, 100_000_000, 2_000_000, 500_000)
    min_score = st.slider("Mindest-Score", 0, 100, 60)
    show_n = st.slider("Anzahl Ergebnisse", 10, 100, 30, 5)
    if st.button("🔄 Daten aktualisieren"):
        st.cache_data.clear()
        st.rerun()

universe = build_universe(universe_choice)
st.info(f"Universe: **{len(universe):,} Aktien/Ticker** (Datenquelle: Yahoo Finance via yfinance)")

with st.spinner(f"Scanne {len(universe):,} Titel …"):
    result = download_prices(tuple(universe))
    if isinstance(result, tuple):
        close, volume = result
        df = calc_scanner(close, volume, min_price, min_dollar)
    else:
        df = pd.DataFrame()

if df.empty:
    st.error("Keine Daten erhalten. Bitte später erneut versuchen.")
    st.stop()

df = df[df["Score"] >= min_score].copy()

# ---------------- Dashboard ----------------
top = df.head(show_n)

c1,c2,c3,c4 = st.columns(4)
c1.metric("🚀 Kandidaten", len(df))
c2.metric("🔥 Score ≥ 80", int((df["Score"] >= 80).sum()))
c3.metric("📈 Ø 5D", f"{top['5D %'].mean():.1f}%")
c4.metric("💥 Ø Volumen", f"{top['Vol x'].mean():.1f}×")

st.subheader("🔥 TOP MOMENTUM")
display = top[["Ticker","Score","Price","1D %","3D %","5D %","20D %","Vol x","RS 5D %","52W High %","Setup"]].copy()
for col in ["Price","1D %","3D %","5D %","20D %","Vol x","RS 5D %","52W High %"]:
    display[col] = display[col].round(2)

st.dataframe(
    display.style.format({
        "Score":"{:.1f}", "Price":"${:.2f}", "1D %":"{:.2f}%", "3D %":"{:.2f}%",
        "5D %":"{:.2f}%", "20D %":"{:.2f}%", "Vol x":"{:.2f}×",
        "RS 5D %":"{:.2f}%", "52W High %":"{:.2f}%"
    }),
    use_container_width=True, hide_index=True
)

st.subheader("🚨 Explosive Setup Watchlist")
explosive = df[
    (df["Score"] >= 75) &
    ((df["Vol x"] >= 2) | (df["5D %"] >= 8) | (df["52W High %"] >= -2))
].head(20)
st.dataframe(
    explosive[["Ticker","Score","5D %","20D %","Vol x","RS 5D %","52W High %","Setup"]]
    .round(2),
    use_container_width=True, hide_index=True
)

st.subheader("📊 Best Relative Strength")
rs = df.sort_values(["RS 5D %","Score"], ascending=False).head(20)
st.dataframe(
    rs[["Ticker","Score","5D %","20D %","RS 5D %","RS 20D %","Vol x","Setup"]].round(2),
    use_container_width=True, hide_index=True
)

st.subheader("💥 Biggest Volume Surges")
vs = df.sort_values(["Vol x","Score"], ascending=False).head(20)
st.dataframe(
    vs[["Ticker","Score","1D %","5D %","Vol x","20D %","Setup"]].round(2),
    use_container_width=True, hide_index=True
)

st.caption(
    f"Letztes Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
    "Technischer Scanner, keine Anlageberatung. Yahoo/yfinance kann verzögerte oder unvollständige Daten liefern."
)
