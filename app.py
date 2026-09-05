import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="US Momentum Scanner",
    page_icon="🚀",
    layout="wide",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #07101d;
    color: #eef4ff;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
}

h1 {
    font-size: 2.5rem !important;
}

.metric-card {
    background: #0d1929;
    border: 1px solid #1d3148;
    border-radius: 14px;
    padding: 14px;
    min-height: 95px;
}

.small {
    color: #91a5bd;
    font-size: .85rem;
}

.good {
    color: #39e58c;
}

.bad {
    color: #ff5d6c;
}

.warn {
    color: #ffd166;
}

.score {
    font-size: 1.7rem;
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

@media (max-width: 700px) {

    h1 {
        font-size: 1.8rem !important;
    }

    .block-container {
        padding: .7rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.title("🚀 US Momentum & Breakout Scanner")

st.caption(
    "Findet US-Aktien mit starkem Momentum, Relative Strength, "
    "Volumenanstieg und Breakout-Signalen."
)


# ============================================================
# S&P 500
# ============================================================

@st.cache_data(ttl=86400)
def get_sp500():

    try:

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )

        df = tables[0]

        tickers = (
            df["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.upper()
            .tolist()
        )

        return sorted(set(tickers))

    except Exception as e:

        st.warning(f"S&P-500-Liste konnte nicht geladen werden: {e}")

        return [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "META",
            "GOOGL",
            "AVGO",
            "TSLA",
            "JPM",
            "LLY",
        ]


# ============================================================
# NASDAQ 100
# ============================================================

@st.cache_data(ttl=86400)
def get_nasdaq100():

    try:

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100"
        )

        for table in tables:

            for column in table.columns:

                if "Ticker" in str(column):

                    tickers = (
                        table[column]
                        .astype(str)
                        .str.replace(".", "-", regex=False)
                        .str.upper()
                        .tolist()
                    )

                    if len(tickers) > 50:

                        return sorted(set(tickers))

    except Exception as e:

        st.warning(
            f"Nasdaq-100-Liste konnte nicht geladen werden: {e}"
        )

    return [
        "NVDA",
        "AMD",
        "AVGO",
        "MU",
        "PLTR",
        "APP",
        "MSTR",
        "NFLX",
        "COST",
        "QCOM",
    ]


# ============================================================
# RUSSELL 2000 / IWM
# ============================================================

@st.cache_data(ttl=86400)
def get_russell2000():

    urls = [

        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax?"
        "fileType=csv&fileName=IWM_holdings&dataType=fund",

        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax?"
        "fileType=csv&fileName=IWM_holdings&dataType=fund"
        "&asOfDate=20260101",

    ]

    for url in urls:

        try:

            response = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )

            response.raise_for_status()

            text = response.content.decode(
                "utf-8",
                errors="ignore",
            )

            text = text.replace("\x00", "")

            lines = text.splitlines()

            header_index = None

            for i, line in enumerate(lines):

                if "Ticker" in line and "Name" in line:

                    header_index = i
                    break

            if header_index is None:
                continue

            csv_text = "\n".join(
                lines[header_index:]
            )

            df = pd.read_csv(
                io.StringIO(csv_text)
            )

            ticker_column = None

            for column in df.columns:

                if str(column).strip().lower() == "ticker":

                    ticker_column = column
                    break

            if ticker_column is None:
                continue

            tickers = (
                df[ticker_column]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)
            )

            tickers = [

                ticker

                for ticker in tickers

                if re.fullmatch(
                    r"[A-Z]{1,5}(?:-[A-Z]{1,2})?",
                    ticker,
                )

            ]

            tickers = sorted(set(tickers))

            if len(tickers) >= 1000:

                return tickers

        except Exception:
            continue

    # Fallback
    return [
        "RKLB",
        "IONQ",
        "SOUN",
        "HIMS",
        "CELH",
        "ASTS",
        "CRDO",
        "TMDX",
        "SMCI",
        "CAVA",
        "RXRX",
        "ENVX",
        "RIVN",
        "JOBY",
        "ACHR",
        "OPEN",
        "LCID",
        "UPST",
        "AFRM",
    ]


# ============================================================
# BUILD UNIVERSE
# ============================================================

def build_universe(choice):

    tickers = set()

    if choice in [
        "S&P 500",
        "S&P 500 + Nasdaq-100",
        "S&P 500 + Nasdaq-100 + Russell 2000",
    ]:

        tickers.update(
            get_sp500()
        )

    if choice in [
        "S&P 500 + Nasdaq-100",
        "S&P 500 + Nasdaq-100 + Russell 2000",
    ]:

        tickers.update(
            get_nasdaq100()
        )

    if choice in [
        "Russell 2000",
        "S&P 500 + Nasdaq-100 + Russell 2000",
    ]:

        tickers.update(
            get_russell2000()
        )

    tickers.update(
        [
            "SPY",
            "QQQ",
        ]
    )

    tickers = [

        ticker

        for ticker in tickers

        if ticker
        and ticker.upper() not in ["NAN", "USD"]

    ]

    return sorted(set(tickers))


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def download_market_data(tickers):

    data = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:

        return None, None

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        close = data["Close"].copy()

        volume = data["Volume"].copy()

    else:

        close = data[
            ["Close"]
        ].copy()

        volume = data[
            ["Volume"]
        ].copy()

    close = close.dropna(
        axis=1,
        how="all",
    )

    volume = volume.reindex(
        columns=close.columns
    )

    return close, volume


# ============================================================
# MOMENTUM SCANNER
# ============================================================

def calculate_scanner(
    close,
    volume,
    minimum_price,
    minimum_dollar_volume,
):

    results = []

    if "SPY" not in close.columns:

        return pd.DataFrame()

    spy = close["SPY"].dropna()

    if len(spy) < 60:

        return pd.DataFrame()

    spy_5 = spy.pct_change(5).iloc[-1]

    spy_20 = spy.pct_change(20).iloc[-1]

    for ticker in close.columns:

        if ticker in [
            "SPY",
            "QQQ",
        ]:

            continue

        price_series = (
            close[ticker]
            .dropna()
        )

        if len(price_series) < 65:

            continue

        volume_series = (
            volume[ticker]
            .reindex(
                price_series.index
            )
            .fillna(0)
        )

        price = float(
            price_series.iloc[-1]
        )

        average_volume_20 = float(
            volume_series
            .tail(20)
            .mean()
        )

        dollar_volume = float(
            (
                price_series.tail(20)
                * volume_series.tail(20)
            ).mean()
        )

        if price < minimum_price:

            continue

        if dollar_volume < minimum_dollar_volume:

            continue

        if average_volume_20 <= 0:

            continue

        # ----------------------------------------------------
        # RETURNS
        # ----------------------------------------------------

        r1 = price_series.pct_change(
            1
        ).iloc[-1]

        r3 = price_series.pct_change(
            3
        ).iloc[-1]

        r5 = price_series.pct_change(
            5
        ).iloc[-1]

        r20 = price_series.pct_change(
            20
        ).iloc[-1]

        r60 = price_series.pct_change(
            60
        ).iloc[-1]

        # ----------------------------------------------------
        # MOVING AVERAGES
        # ----------------------------------------------------

        ma20 = (
            price_series
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        ma50 = (
            price_series
            .rolling(50)
            .mean()
            .iloc[-1]
        )

        if len(price_series) >= 200:

            ma200 = (
                price_series
                .rolling(200)
                .mean()
                .iloc[-1]
            )

        else:

            ma200 = np.nan

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        volume_ratio = float(
            volume_series.iloc[-1]
            / average_volume_20
        )

        # ----------------------------------------------------
        # HIGH LEVELS
        # ----------------------------------------------------

        high20 = (
            price_series
            .tail(20)
            .max()
        )

        high52 = (
            price_series
            .tail(252)
            .max()
        )

        distance_52w = (
            price / high52
        ) - 1

        breakout20 = (
            price >= high20 * 0.995
        )

        near52 = (
            price >= high52 * 0.97
        )

        # ----------------------------------------------------
        # MOMENTUM ACCELERATION
        # ----------------------------------------------------

        acceleration = (
            r5 - (r20 / 4)
        )

        # ----------------------------------------------------
        # RELATIVE STRENGTH
        # ----------------------------------------------------

        rs5 = r5 - spy_5

        rs20 = r20 - spy_20

        # ----------------------------------------------------
        # TREND SCORE
        # ----------------------------------------------------

        trend_score = 0

        if price > ma20:

            trend_score += 25

        if price > ma50:

            trend_score += 25

        if not np.isnan(ma200):

            if price > ma200:

                trend_score += 25

        if ma20 > ma50:

            trend_score += 25

        # ----------------------------------------------------
        # MOMENTUM SCORE
        # ----------------------------------------------------

        momentum_score = np.clip(

            0.35 * (
                (r5 + 0.10)
                / 0.20
            )

            +

            0.40 * (
                (r20 + 0.20)
                / 0.40
            )

            +

            0.25 * (
                (r60 + 0.30)
                / 0.60
            ),

            0,
            1,

        ) * 100

        # ----------------------------------------------------
        # RELATIVE STRENGTH SCORE
        # ----------------------------------------------------

        rs_score = np.clip(

            0.55 * (
                (rs5 + 0.08)
                / 0.16
            )

            +

            0.45 * (
                (rs20 + 0.15)
                / 0.30
            ),

            0,
            1,

        ) * 100

        # ----------------------------------------------------
        # VOLUME SCORE
        # ----------------------------------------------------

        volume_score = np.clip(

            (volume_ratio - 0.8)
            / 3.2,

            0,
            1,

        ) * 100

        # ----------------------------------------------------
        # BREAKOUT SCORE
        # ----------------------------------------------------

        breakout_score = 25

        if breakout20:

            breakout_score += 45

        if near52:

            breakout_score += 30

        # ----------------------------------------------------
        # ACCELERATION SCORE
        # ----------------------------------------------------

        acceleration_score = np.clip(

            (acceleration + 0.04)
            / 0.12,

            0,
            1,

        ) * 100

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        score = (

            0.25 * momentum_score

            +

            0.20 * rs_score

            +

            0.20 * volume_score

            +

            0.15 * breakout_score

            +

            0.10 * trend_score

            +

            0.10 * acceleration_score

        )

        # ----------------------------------------------------
        # SETUP LABELS
        # ----------------------------------------------------

        setups = []

        if breakout20:

            setups.append(
                "20D Breakout"
            )

        if near52:

            setups.append(
                "52W High"
            )

        if volume_ratio >= 2:

            setups.append(
                "Volume Surge"
            )

        if acceleration > 0.03:

            setups.append(
                "Acceleration"
            )

        if price > ma20 > ma50:

            setups.append(
                "Trend"
            )

        if rs5 > 0.03:

            setups.append(
                "Strong RS"
            )

        if not setups:

            setups.append(
                "Momentum"
            )

        results.append(

            {
                "Ticker": ticker,

                "Score": round(
                    score,
                    1,
                ),

                "Price": price,

                "1D %": r1 * 100,

                "3D %": r3 * 100,

                "5D %": r5 * 100,

                "20D %": r20 * 100,

                "60D %": r60 * 100,

                "Vol x": volume_ratio,

                "RS 5D %": rs5 * 100,

                "RS 20D %": rs20 * 100,

                "52W High %":
                    distance_52w * 100,

                "Dollar Vol $":
                    dollar_volume,

                "Setup":
                    " • ".join(
                        setups
                    ),
            }

        )

    if not results:

        return pd.DataFrame()

    result = pd.DataFrame(
        results
    )

    return result.sort_values(
        "Score",
        ascending=False,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Scanner")

    universe_choice = st.selectbox(

        "Aktien-Universum",

        [

            "S&P 500 + Nasdaq-100 + Russell 2000",

            "S&P 500 + Nasdaq-100",

            "S&P 500",

            "Russell 2000",

        ],

    )

    minimum_price = st.number_input(

        "Mindestpreis ($)",

        min_value=0.0,

        max_value=500.0,

        value=5.0,

        step=1.0,

    )

    minimum_dollar_volume = st.number_input(

        "Min. Ø Tagesvolumen ($)",

        min_value=0,

        max_value=100_000_000,

        value=2_000_000,

        step=500_000,

    )

    minimum_score = st.slider(

        "Mindest-Score",

        0,

        100,

        60,

    )

    number_results = st.slider(

        "Anzahl Ergebnisse",

        10,

        100,

        30,

        5,

    )

    if st.button(
        "🔄 Daten aktualisieren"
    ):

        st.cache_data.clear()

        st.rerun()


# ============================================================
# UNIVERSE
# ============================================================

universe = build_universe(
    universe_choice
)

st.info(
    f"Universe: **{len(universe):,} Aktien/Ticker**  "
    f"(Yahoo Finance via yfinance)"
)


# ============================================================
# RUSSELL WARNING
# ============================================================

if (
    "Russell 2000"
    in universe_choice
    and len(universe) < 1000
):

    st.warning(

        "⚠️ Der Russell-2000-Import hat nicht die "
        "erwartete Größe erreicht. Die öffentliche "
        "IWM-Holdings-Datei ist momentan nicht "
        "erreichbar. Der Scanner verwendet deshalb "
        "eine Fallback-Liste. S&P 500 und Nasdaq-100 "
        "werden weiterhin separat geladen."

    )


# ============================================================
# SCAN
# ============================================================

with st.spinner(
    f"Scanne {len(universe):,} Aktien ..."
):

    close, volume = (
        download_market_data(
            tuple(universe)
        )
    )

    if close is None:

        st.error(
            "Keine Marktdaten erhalten."
        )

        st.stop()

    scanner = calculate_scanner(

        close,

        volume,

        minimum_price,

        minimum_dollar_volume,

    )


if scanner.empty:

    st.error(
        "Keine geeigneten Aktien gefunden."
    )

    st.stop()


# ============================================================
# FILTER
# ============================================================

filtered = scanner[
    scanner["Score"]
    >= minimum_score
].copy()

top = filtered.head(
    number_results
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🚀 Kandidaten",
    len(filtered),
)

c2.metric(
    "🔥 Score ≥ 80",
    int(
        (
            filtered["Score"]
            >= 80
        ).sum()
    ),
)

if not top.empty:

    c3.metric(
        "📈 Ø 5D",
        f"{top['5D %'].mean():.1f}%",
    )

    c4.metric(
        "💥 Ø Volumen",
        f"{top['Vol x'].mean():.1f}×",
    )


# ============================================================
# TOP MOMENTUM
# ============================================================

st.subheader(
    "🔥 TOP MOMENTUM"
)

display = top[
    [
        "Ticker",
        "Score",
        "Price",
        "1D %",
        "3D %",
        "5D %",
        "20D %",
        "Vol x",
        "RS 5D %",
        "52W High %",
        "Setup",
    ]
].copy()

display = display.round(
    2
)

st.dataframe(

    display,

    use_container_width=True,

    hide_index=True,

)


# ============================================================
# EXPLOSIVE SETUPS
# ============================================================

st.subheader(
    "🚨 EXPLOSIVE SETUPS"
)

explosive = filtered[
    (filtered["Score"] >= 75)

    &

    (
        (filtered["Vol x"] >= 2)

        |

        (filtered["5D %"] >= 8)

        |

        (filtered["52W High %"] >= -2)
    )
].head(20)

st.dataframe(

    explosive[
        [
            "Ticker",
            "Score",
            "5D %",
            "20D %",
            "Vol x",
            "RS 5D %",
            "52W High %",
            "Setup",
        ]
    ].round(2),

    use_container_width=True,

    hide_index=True,

)


# ============================================================
# RELATIVE STRENGTH
# ============================================================

st.subheader(
    "🏆 BEST RELATIVE STRENGTH"
)

relative_strength = (

    filtered

    .sort_values(
        [
            "RS 5D %",
            "Score",
        ],
        ascending=False,
    )

    .head(20)

)

st.dataframe(

    relative_strength[
        [
            "Ticker",
            "Score",
            "5D %",
            "20D %",
            "RS 5D %",
            "RS 20D %",
            "Vol x",
            "Setup",
        ]
    ].round(2),

    use_container_width=True,

    hide_index=True,

)


# ============================================================
# VOLUME SURGES
# ============================================================

st.subheader(
    "💥 BIGGEST VOLUME SURGES"
)

volume_surges = (

    filtered

    .sort_values(
        [
            "Vol x",
            "Score",
        ],
        ascending=False,
    )

    .head(20)

)

st.dataframe(

    volume_surges[
        [
            "Ticker",
            "Score",
            "1D %",
            "5D %",
            "Vol x",
            "20D %",
            "Setup",
        ]
    ].round(2),

    use_container_width=True,

    hide_index=True,

)


# ============================================================
# ALL SCANNED STOCKS
# ============================================================

with st.expander(
    "📋 Alle gescannten Aktien anzeigen"
):

    st.dataframe(

        scanner.round(2),

        use_container_width=True,

        hide_index=True,

    )


# ============================================================
# FOOTER
# ============================================================

st.caption(

    f"Letztes Update: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "

    "Technischer Scanner, keine Anlageberatung. "
    "Yahoo Finance/yfinance kann verzögerte oder "
    "unvollständige Daten liefern."

)
