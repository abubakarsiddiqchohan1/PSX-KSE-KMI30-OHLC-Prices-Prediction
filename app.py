# =====================================================================
#  PSX PulseCast  -  AI price forecasts for KSE-30 & KMI-30 stocks
#  Developed by Abu Bakar Siddiq
# =====================================================================
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import psx_core as core

warnings.filterwarnings("ignore")

APP_NAME = "PSX PulseCast"
PORTFOLIO_URL = "https://abubakarsiddiq.netlify.app"
MODEL_FILE = "psx_model.pkl"
FIELDS = core.FIELDS
COLORS = {"Open": "#2563eb", "High": "#16a34a", "Low": "#dc2626", "Close": "#f59e0b"}
BULL, BEAR = "#16a34a", "#dc2626"
SIGNAL_STYLE = {"BUY": ("#dcfce7", "#166534"), "SELL": ("#fee2e2", "#991b1b"), "HOLD": ("#e5e7eb", "#374151")}

st.set_page_config(page_title=f"{APP_NAME} | PSX Price Forecasts", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 5rem; max-width: 1500px;}
.hero {background: linear-gradient(120deg,#0b1220 0%,#0e4f6b 55%,#14b8a6 100%);
       padding: 26px 32px; border-radius: 18px; margin-bottom: 18px; color: #fff;
       box-shadow: 0 8px 24px rgba(14,79,107,.25);}
.hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: .3px; color:#fff;}
.hero p {margin: 6px 0 0 0; font-size: 1.02rem; opacity: .92; color:#e6fffb;}
.card {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 18px;
       box-shadow: 0 2px 8px rgba(15,23,42,.05); height: 100%;}
.card .lbl {font-size: .78rem; text-transform: uppercase; letter-spacing: .8px; color: #6b7280; font-weight: 600;}
.card .val {font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1.25;}
.card .sub {font-size: .82rem; color: #6b7280;}
.sec {font-size: 1.35rem; font-weight: 800; color: #0f172a; margin: 22px 0 8px 0;
      padding-left: 12px; border-left: 5px solid #14b8a6;}
.badge {display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem; font-weight: 700;
        background:#ccfbf1; color:#115e59; margin-left: 8px; vertical-align: middle;}
.empty {text-align:center; padding: 60px 20px; border: 2px dashed #cbd5e1; border-radius: 16px;
        color:#475569; background:#f8fafc; font-size:1.1rem;}
.footer {position: fixed; left: 0; right: 0; bottom: 0; text-align: center; padding: 9px 10px;
         background: #0b1220; color: #cbd5e1; font-size: .92rem; z-index: 999;}
.footer a {color: #5eead4; font-weight: 700; text-decoration: none;}
.footer a:hover {text-decoration: underline;}
section[data-testid="stSidebar"] {background: #f1f5f9;}
div.stButton > button {border-radius: 10px; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", color="#0f172a"):
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>')


# ------------------------------------------------------------- load model
@st.cache_resource(show_spinner="Loading model…")
def load_bundle():
    here = Path(__file__).resolve().parent
    for p in (here / MODEL_FILE, Path.cwd() / MODEL_FILE):
        if p.exists():
            b = joblib.load(p)
            h = pd.DataFrame(b["history"])
            h["Date"] = pd.to_datetime(h["Date"])
            b["hist"] = h.sort_values(["Symbol", "Date"]).reset_index(drop=True)
            return b
    return None


BUNDLE = load_bundle()
if BUNDLE is None:
    st.error(f"Model file **{MODEL_FILE}** was not found next to app.py. Train the model first (Cell 3).")
    st.stop()

HIST = BUNDLE["hist"]
COMPANIES = sorted(BUNDLE["companies"], key=lambda c: c["name"])
NAMES = {c["Symbol"]: c["name"] for c in COMPANIES}
INDEX = {c["Symbol"]: c["index"] for c in COMPANIES}
SYMBOLS = list(NAMES.keys())
LAST_DATE = HIST["Date"].max().date()
DEFAULT_START = (pd.Timestamp(LAST_DATE) + pd.offsets.BDay(1)).date()
DEFAULT_DAYS = 5


def company_hist(symbol):
    return core.add_indicators(HIST[HIST["Symbol"] == symbol][["Date"] + FIELDS])


@st.cache_data(show_spinner=False, max_entries=128)
def run_forecast(symbol, start_iso, n):
    g = HIST[HIST["Symbol"] == symbol][["Date"] + FIELDS]
    cut, hidden, shown = core.plan_dates(g["Date"], start_iso, n)
    targets = hidden + shown
    return core.forecast(g.iloc[:cut], BUNDLE["models"], BUNDLE["feature_cols"], targets, show_from=len(hidden))


# ------------------------------------------------------------ session state
st.session_state.setdefault("fields", list(FIELDS))
st.session_state.setdefault("days", DEFAULT_DAYS)
st.session_state.setdefault("start", DEFAULT_START)
st.session_state.setdefault("go", False)
st.session_state.setdefault("style", "Candlestick (OHLC)")


def clear_company():
    st.session_state["company"] = None
    st.session_state["go"] = False


def reset_all():
    st.session_state["company"] = None
    st.session_state["fields"] = list(FIELDS)
    st.session_state["days"] = DEFAULT_DAYS
    st.session_state["start"] = DEFAULT_START
    st.session_state["style"] = "Candlestick (OHLC)"
    st.session_state["go"] = False


def company_changed():
    st.session_state["go"] = False


def start_predict():
    st.session_state["go"] = True


# ----------------------------------------------------------------- charts
PLOT_CFG = {"displaylogo": False, "scrollZoom": True, "displayModeBar": True,
            "toImageButtonOptions": {"format": "png", "scale": 2, "height": 800, "width": 1600}}


def hist_chart(d, symbol, name, style, fields):
    fig = go.Figure()
    dates = pd.DatetimeIndex(d["Date"])
    all_days = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = [x.strftime("%Y-%m-%d") for x in all_days.difference(dates)]

    # bullish / bearish zones (MA9 vs MA21)
    shapes, seg_start, seg_trend = [], None, None
    rows = d[d["Trend"] != ""].reset_index(drop=True)
    for i, r in rows.iterrows():
        if r["Trend"] != seg_trend:
            if seg_trend is not None:
                shapes.append((seg_start, r["Date"], seg_trend))
            seg_start, seg_trend = r["Date"], r["Trend"]
    if seg_trend is not None:
        shapes.append((seg_start, rows["Date"].iloc[-1] + pd.Timedelta(days=1), seg_trend))
    fig.update_layout(shapes=[dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1,
                                   fillcolor=(BULL if t == "Bullish" else BEAR), opacity=0.09,
                                   layer="below", line_width=0) for a, b, t in shapes])

    if style.startswith("Candlestick"):
        fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                     name="Price (O-H-L-C)", increasing_line_color=BULL,
                                     decreasing_line_color=BEAR, increasing_fillcolor=BULL,
                                     decreasing_fillcolor=BEAR))
    else:
        for f in fields:
            fig.add_trace(go.Scatter(x=d["Date"], y=d[f], mode="lines+markers", name=f,
                                     line=dict(color=COLORS[f], width=2), marker=dict(size=4),
                                     hovertemplate=f + ": %{y:,.2f}<extra></extra>"))

    ma_style = {"MA5": ("#7c3aed", "solid", 1.6), "MA9": ("#0891b2", "solid", 1.6),
                "MA21": ("#ea580c", "solid", 1.8), "MA200": ("#111827", "dash", 2.2)}
    for k, (col, dash, w) in ma_style.items():
        if d[k].notna().any():
            fig.add_trace(go.Scatter(x=d["Date"], y=d[k], mode="lines", name=f"{k[2:]}-Day MA",
                                     line=dict(color=col, dash=dash, width=w),
                                     hovertemplate=f"{k[2:]}-Day MA: " + "%{y:,.2f}<extra></extra>"))

    # crossovers
    tr = d["Trend"].values
    up = [i for i in range(1, len(d)) if tr[i] == "Bullish" and tr[i - 1] == "Bearish"]
    dn = [i for i in range(1, len(d)) if tr[i] == "Bearish" and tr[i - 1] == "Bullish"]
    if up:
        fig.add_trace(go.Scatter(x=d["Date"].iloc[up], y=d["Low"].iloc[up] * 0.99, mode="markers",
                                 name="Bullish crossover", marker=dict(symbol="triangle-up", size=13, color=BULL,
                                                                       line=dict(color="white", width=1)),
                                 hovertemplate="Bullish crossover (MA9 above MA21)<extra></extra>"))
    if dn:
        fig.add_trace(go.Scatter(x=d["Date"].iloc[dn], y=d["High"].iloc[dn] * 1.01, mode="markers",
                                 name="Bearish crossover", marker=dict(symbol="triangle-down", size=13, color=BEAR,
                                                                       line=dict(color="white", width=1)),
                                 hovertemplate="Bearish crossover (MA9 below MA21)<extra></extra>"))
    for lbl, col in (("Bullish zone (MA9 > MA21)", BULL), ("Bearish zone (MA9 < MA21)", BEAR)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lbl,
                                 marker=dict(symbol="square", size=13, color=col, opacity=0.35)))

    fig.update_layout(
        title=dict(text=f"<b>{symbol} — {name}</b><br><sup>Daily Open · High · Low · Close with 5, 9, 21 &amp; 200-Day "
                        f"Moving Averages and Bullish / Bearish Market Zones</sup>", x=0.01),
        xaxis=dict(title="Date", rangeslider=dict(visible=True, thickness=0.07), type="date",
                   rangebreaks=[dict(values=missing)], hoverformat="%a, %d %b %Y", showspikes=True,
                   rangeselector=dict(buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                                               dict(count=3, label="3M", step="month", stepmode="backward"),
                                               dict(count=6, label="6M", step="month", stepmode="backward"),
                                               dict(step="all", label="All")],
                                      bgcolor="#e0f2fe", activecolor="#14b8a6", y=1.02)),
        yaxis=dict(title="Price (PKR)", tickformat=",.0f", fixedrange=False, side="left"),
        hovermode="x unified", template="plotly_white", height=720, margin=dict(t=110, b=40, l=60, r=30),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, title_text="Legend: "),
        xaxis_rangeslider_yaxis_rangemode="match")
    return fig


def forecast_chart(fc, past, ctx, symbol, name, fields, start_label):
    fig = go.Figure()
    for f in fields:
        # actual line = context before forecast + actual days inside the window (if they already exist)
        act = pd.concat([ctx[["Date", f]], past[["Date", f]]]).drop_duplicates("Date").sort_values("Date")
        fig.add_trace(go.Scatter(x=act["Date"], y=act[f], mode="lines+markers", name=f"{f} — Actual",
                                 line=dict(color=COLORS[f], width=1.6, dash="dot"), marker=dict(size=4),
                                 opacity=0.65, hovertemplate=f"{f} actual: " + "%{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc[f], mode="lines+markers", name=f"{f} — Predicted",
                                 line=dict(color=COLORS[f], width=3), marker=dict(size=8, symbol="diamond"),
                                 hovertemplate=f"{f} predicted: " + "%{y:,.2f}<extra></extra>"))
    anchor = "Close" if "Close" in fields else fields[0]
    for sig, sym, col in (("BUY", "triangle-up", BULL), ("SELL", "triangle-down", BEAR), ("HOLD", "circle", "#6b7280")):
        m = fc[fc["Signal"] == sig]
        if len(m):
            fig.add_trace(go.Scatter(
                x=m["Date"], y=m[anchor] * (1.004 if sig == "BUY" else 0.996 if sig == "SELL" else 1.0),
                mode="markers", name=f"{sig} signal",
                marker=dict(symbol=sym, size=15 if sig != "HOLD" else 9, color=col, line=dict(color="white", width=1)),
                customdata=np.c_[m["Expected Change %"]],
                hovertemplate=sig + " · expected change %{customdata[0]:+.2f}%<extra></extra>"))
    x0 = fc["Date"].min().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=x0, x1=x0, y0=0, y1=1, yref="paper", line=dict(color="#64748b", dash="dash", width=1.5))
    fig.add_annotation(x=x0, y=1.0, yref="paper", text="Forecast start", showarrow=False, yshift=12,
                       font=dict(color="#475569"))
    fig.update_layout(
        title=dict(text=f"<b>{symbol} — {name}</b><br><sup>Predicted Prices ({len(fc)} trading days from {start_label}) "
                        f"with BUY / SELL / HOLD signals</sup>", x=0.01),
        xaxis=dict(title="Date", type="date", rangebreaks=[dict(bounds=["sat", "mon"])],
                   hoverformat="%a, %d %b %Y", showspikes=True),
        yaxis=dict(title="Price (PKR)", tickformat=",.0f"),
        hovermode="x unified", template="plotly_white", height=620, margin=dict(t=110, b=40, l=60, r=30),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title_text="Legend: "))
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"### 📈 {APP_NAME}")
    st.markdown("**1 · Company**")
    c1, c2 = st.columns([5, 1.3])
    c1.selectbox("Company", SYMBOLS, index=None, key="company", placeholder="Type to search a company…",
                 format_func=lambda s: f"{s} — {NAMES[s]}", label_visibility="collapsed", on_change=company_changed)
    c2.button("✖", key="btn_clear", help="Clear company", on_click=clear_company, width="stretch")
    company = st.session_state.get("company")

    st.markdown("**2 · Price fields**")
    st.multiselect("Fields", FIELDS, key="fields", label_visibility="collapsed")

    st.markdown("**3 · Days to predict**")
    st.slider("Days", 1, 30, key="days", label_visibility="collapsed")

    st.markdown("**4 · Prediction start date**")
    if company:
        dts = HIST.loc[HIST["Symbol"] == company, "Date"].reset_index(drop=True)
    else:
        dts = HIST.drop_duplicates("Date")["Date"].sort_values().reset_index(drop=True)
    min_d = dts.iloc[core.MIN_HISTORY].date()
    max_d = LAST_DATE + timedelta(days=90)
    st.session_state["start"] = min(max(st.session_state["start"], min_d), max_d)
    st.date_input("Start date", min_value=min_d, max_value=max_d, key="start", label_visibility="collapsed",
                  format="DD/MM/YYYY")
    st.caption("Pick a past date to test the model against real prices, or a future date to forecast.")

    st.button("🔮  Predict", type="primary", width="stretch", on_click=start_predict,
              disabled=(not company or not st.session_state["fields"]))
    st.button("🔄  Reset / Clear All", width="stretch", on_click=reset_all)

# ---------------------------------------------------------------------- main
st.markdown(f"""<div class="hero"><h1>📈 {APP_NAME}</h1>
<p>AI-powered Open · High · Low · Close forecasts with Buy / Sell signals for the KSE-30 &amp; KMI-30 stocks on the Pakistan Stock Exchange</p></div>""",
            unsafe_allow_html=True)

if not company:
    st.markdown('<div class="empty">👈 Select a company from the sidebar to see its price history and forecasts.</div>',
                unsafe_allow_html=True)
else:
    name = NAMES[company]
    d = company_hist(company)
    fields = st.session_state["fields"] or list(FIELDS)
    last, prev = d.iloc[-1], d.iloc[-2]
    chg = (last["Close"] / prev["Close"] - 1) * 100
    trend = last["Trend"] or "—"

    st.markdown(f"## {company} · {name} <span class='badge'>{INDEX[company]}</span>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(card("Last close", f"PKR {last['Close']:,.2f}", last["Date"].strftime("%a, %d %b %Y")), unsafe_allow_html=True)
    k2.markdown(card("Day change", f"{chg:+.2f}%", f"vs PKR {prev['Close']:,.2f}", BULL if chg >= 0 else BEAR),
                unsafe_allow_html=True)
    k3.markdown(card("Market trend", trend, "9-day MA vs 21-day MA", BULL if trend == "Bullish" else BEAR if trend == "Bearish" else "#0f172a"),
                unsafe_allow_html=True)
    k4.markdown(card("Range (all data)", f"{d['Low'].min():,.0f} – {d['High'].max():,.0f}", "Lowest low – highest high (PKR)"),
                unsafe_allow_html=True)

    # ---------------------------------------------------- historical section
    st.markdown('<div class="sec">Historical Prices</div>', unsafe_allow_html=True)
    st.radio("Chart style", ["Candlestick (OHLC)", "Lines (selected fields)"], key="style", horizontal=True,
             label_visibility="collapsed")
    st.plotly_chart(hist_chart(d, company, name, st.session_state["style"], fields), width="stretch", config={
        **PLOT_CFG, "toImageButtonOptions": {**PLOT_CFG["toImageButtonOptions"], "filename": f"{company}_historical"}},
        key=f"hist_{company}")
    with st.expander("📅 Daily historical data (every date)"):
        tbl = d.sort_values("Date", ascending=False).copy()
        tbl["Date"] = tbl["Date"].dt.strftime("%a, %d %b %Y")
        tbl = tbl.rename(columns={"MA5": "5-Day MA", "MA9": "9-Day MA", "MA21": "21-Day MA", "MA200": "200-Day MA"})
        st.dataframe(tbl.round(2), hide_index=True, width="stretch", height=380)

    # ----------------------------------------------------- forecast section
    st.markdown('<div class="sec">Price Forecast</div>', unsafe_allow_html=True)
    if not st.session_state["go"]:
        st.info("Choose your fields, number of days and start date, then press **🔮 Predict**.")
    else:
        try:
            with st.spinner("Forecasting…"):
                fc = run_forecast(company, pd.Timestamp(st.session_state["start"]).strftime("%Y-%m-%d"),
                                  int(st.session_state["days"]))
        except ValueError as e:
            st.warning(str(e))
            st.stop()

        comp = HIST[HIST["Symbol"] == company]
        past = comp[comp["Date"].isin(fc["Date"])]
        ctx = comp[comp["Date"] < fc["Date"].min()].tail(20)
        start_label = fc["Date"].min().strftime("%d %b %Y")

        first = fc.iloc[0]
        counts = fc["Signal"].value_counts()
        mix = " · ".join(f"{counts.get(s, 0)} {s}" for s in ("BUY", "SELL", "HOLD"))
        scol = {"BUY": BULL, "SELL": BEAR, "HOLD": "#6b7280"}[first["Signal"]]
        anchor = "Close" if "Close" in fields else fields[0]

        # table (selected fields only)
        out = pd.DataFrame({"Date": fc["Date"]})
        for f in fields:
            out[f"{f} (Predicted)"] = fc[f]
        out["Expected Change %"] = fc["Expected Change %"]
        out["Signal"] = fc["Signal"]
        merged = fc[["Date"] + fields].merge(past[["Date"] + fields], on="Date", how="left", suffixes=("", "_act"))
        has_actual = merged[[f"{f}_act" for f in fields]].notna().any(axis=1)
        avg_err = None
        if has_actual.any():
            for f in fields:
                out[f"{f} (Actual)"] = merged[f"{f}_act"].values
            errs = np.column_stack([np.abs(merged[f] / merged[f"{f}_act"] - 1) * 100 for f in fields])
            out["Avg Error %"] = np.nanmean(errs, axis=1)
            avg_err = float(np.nanmean(errs))

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(card("Forecast window", f"{len(fc)} day{'s' if len(fc) > 1 else ''}",
                         f"{fc['Date'].min().strftime('%d %b')} → {fc['Date'].max().strftime('%d %b %Y')}"), unsafe_allow_html=True)
        m2.markdown(card("First-day signal", first["Signal"], f"Expected change {first['Expected Change %']:+.2f}%", scol),
                    unsafe_allow_html=True)
        m3.markdown(card(f"Predicted {anchor} (last day)", f"PKR {fc[anchor].iloc[-1]:,.2f}",
                         fc["Date"].iloc[-1].strftime("%a, %d %b %Y")), unsafe_allow_html=True)
        if avg_err is not None:
            m4.markdown(card("Avg error vs actual", f"{avg_err:.2f}%", f"{int(has_actual.sum())} day(s) compared"),
                        unsafe_allow_html=True)
        else:
            m4.markdown(card("Signal mix", mix, "BUY · SELL · HOLD days"), unsafe_allow_html=True)

        st.plotly_chart(forecast_chart(fc, past, ctx, company, name, fields, start_label), width="stretch",
                        config={**PLOT_CFG, "toImageButtonOptions": {**PLOT_CFG["toImageButtonOptions"],
                                                                      "filename": f"{company}_forecast"}},
                        key=f"fc_{company}")

        st.markdown("**Daily predictions**")
        show = out.copy()
        show["Date"] = show["Date"].dt.strftime("%a, %d %b %Y")
        num_cols = [c for c in show.columns if c not in ("Date", "Signal", "Expected Change %", "Avg Error %")]
        fmt = {c: "{:,.2f}" for c in num_cols}
        fmt.update({"Expected Change %": "{:+.2f}%", "Avg Error %": "{:.2f}%"})
        sty = (show.style.format(fmt, na_rep="—")
               .map(lambda v: f"background-color:{SIGNAL_STYLE[v][0]};color:{SIGNAL_STYLE[v][1]};font-weight:700;text-align:center",
                    subset=["Signal"]))
        st.dataframe(sty, hide_index=True, width="stretch")

        csv_df = out.copy()
        csv_df.insert(0, "Company", name)
        csv_df.insert(0, "Symbol", company)
        csv_df["Date"] = csv_df["Date"].dt.strftime("%Y-%m-%d")
        csv_df = csv_df.round(2)
        csv_bytes = csv_df.to_csv(index=False)
        b1, b2 = st.columns([1, 3])
        b1.download_button("⬇️ Download CSV", csv_bytes, file_name=f"{company}_forecast_{fc['Date'].min():%Y%m%d}_{len(fc)}d.csv",
                           mime="text/csv", width="stretch")
        with st.expander("📋 Copy predictions as text"):
            st.code(csv_bytes, language="csv")

# ------------------------------------------------------------------- footer
st.markdown(f'<div class="footer">Developed by <a href="{PORTFOLIO_URL}" target="_blank">Abu Bakar Siddiq</a> · '
            f'<a href="{PORTFOLIO_URL}" target="_blank">View my portfolio ↗</a> · '
            f'Forecasts are model estimates, not financial advice.</div>', unsafe_allow_html=True)
