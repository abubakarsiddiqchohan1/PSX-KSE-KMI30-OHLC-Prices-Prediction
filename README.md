# 📈 PSX PulseCast

**AI-powered Open · High · Low · Close forecasts with Buy / Sell / Hold signals for the KSE-30 & KMI-30 stocks on the Pakistan Stock Exchange.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3f4f75)
![Streamlit](https://img.shields.io/badge/Streamlit-GUI-ff4b4b)

🔗 **Live demo:** https://psx-kse-kmi30-ohlc-prices-prediction.streamlit.app/

---

## ✨ Features

- **40 companies** common to both the KSE-30 and KMI-30 indices, selectable from a searchable dropdown with a one-click clear button.
- **Forecast any price field:** Open, High, Low, Close, or all four, for 1–30 trading days.
- **Any start date:** predict the future, or pick a past date to compare predictions with real prices and see the average error.
- **Buy / Sell / Hold signal for every predicted date**, based on the predicted move, short-term trend and RSI.
- **Interactive Plotly charts**, each with Reset axes, zoom, pan and Save as PNG:
  - Historical candlestick (or line) chart with every date, 1M / 3M / 6M / All range buttons and a range slider.
  - 5, 9, 21 and 200-day moving averages.
  - Bullish and bearish market zones with crossover markers.
  - A separate predicted-vs-actual forecast chart.
- **Export:** download predictions as CSV or copy them as text.
- **One-click Reset / Clear All** for company, fields, graphs and predictions.

## 🧠 How it works

| Part | Details |
|---|---|
| Data | Daily OHLC for 40 PSX stocks (company, date, symbol, open, high, low, close, index membership) |
| Features | Lagged returns, candle shape, distance from 5/9/21/200-day MAs, volatility, RSI, momentum, day of week |
| Targets | Next-day Open, High, Low and Close as log-returns versus the previous close |
| Model | Ensemble of **HistGradientBoosting + ExtraTrees + Ridge** per target, hyper-parameters tuned with time-series cross-validation |
| Multi-day forecast | Recursive: each predicted day is fed back in to forecast the next one |
| Signal | Predicted move + MA5/MA21 trend + RSI → BUY / SELL / HOLD |
| Regime | **Bullish** when the 9-day MA is above the 21-day MA, otherwise **Bearish** |

### Model accuracy (hold-out, last 40 trading days, 1 day ahead)

| Field | MAPE | Compared with "tomorrow = today's close" |
|---|---|---|
| Open | ~0.47% | about equal |
| High | ~1.04% | clearly better (~27% lower error) |
| Low | ~1.05% | clearly better (~27% lower error) |
| Close | ~1.31% | about equal |

Close direction accuracy is roughly 53%. Daily stock moves are very noisy, so open and close are hard to beat a naive baseline on, while the day's high and low are more predictable. Exact figures are printed in the Colab notebook after training.

> ⚠️ **Disclaimer:** This project is for education and research. Forecasts are model estimates, **not financial advice**. Do not make investment decisions based only on this tool.

## 🗂️ Repository structure

```
├── app.py                 # Streamlit GUI
├── psx_core.py            # Features, ensemble model, forecasting, signals
├── psx_model.pkl          # Trained model + price history
├── requirements.txt       # Dependencies (versions match the saved model)
├── .streamlit/
│   └── config.toml        # Theme (optional)
└── README.md
```

## 🚀 Run locally

```bash
git clone https://github.com/YOUR-USERNAME/psx-pulsecast.git
cd psx-pulsecast
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub, with `app.py` in the root folder.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repo, branch `main`, main file `app.py`.
4. Click **Deploy**.

## 🔄 Updating with new market data

1. Add the new trading days at the bottom of your Excel dataset (same columns).
2. Re-run the training cell in the Colab notebook. It retrains the model and creates a new `psx_model.pkl`.
3. Replace `psx_model.pkl` in this repository. Streamlit Cloud redeploys automatically.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit · joblib

## 👤 Author

**Abu Bakar Siddiq**, Data Science graduate (UMT, Lahore) focused on AI / ML engineering.

- 🌐 Portfolio: [abubakarsiddiq.netlify.app](https://abubakarsiddiq.netlify.app)
- 💼 LinkedIn: [linkedin.com/in/abubakarsiddiqchohan](https://www.linkedin.com/in/abubakarsiddiqchohan)
- 💻 GitHub: [github.com/abubakarsiddiqchohan1](https://github.com/abubakarsiddiqchohan1)

⭐ If you find this project useful, please give it a star.
