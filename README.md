# Trader Performance vs. Market Sentiment Analysis 📈

Analyzes retail crypto trader behavior and performance across different market sentiment regimes (Fear vs. Greed). Features a robust data preprocessing pipeline, behavioral segmentation via clustering, and an interactive Streamlit dashboard for exploring actionable trading strategies.

---

## 📂 Project Structure

```
trader-sentiment-analysis/
├── data/
│   ├── raw/               # Initial raw datasets (historical_data.csv, fear_greed_index.csv)
│   └── processed/         # Cleaned, merged, and engineered datasets
├── src/                   # Core Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   ├── models.py
│   └── utils.py
├── analysis.ipynb         # End-to-end workflow: data quality, modeling, clustering
├── streamlit_app.py       # Interactive dashboard
└── requirements.txt
```

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.9+

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd trader-sentiment-analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
altair
joblib
```

---

## 🚀 How to Run

### Step 1 — Execute the Data Pipeline

Run the Jupyter Notebook to process raw data, generate features, and train clustering models.

```bash
jupyter notebook
```

Open `analysis.ipynb` and **Run All Cells**.

Verify that `merged_data.csv` is generated in `data/processed/`.

### Step 2 — Launch the Interactive Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Methodology

| Stage | Description |
|---|---|
| **Data Alignment & Cleaning** | Trades converted from Unix timestamps to daily dates and aligned with the daily Fear & Greed Index. Missing values handled; trade direction strings standardized to calculate Long/Short ratios. |
| **Feature Engineering** | Leverage (absent from raw data) was simulated via a log-normal distribution skewed right to mimic typical retail behavior. Enables realistic risk evaluation and strategy development. |
| **Metric Aggregation** | Calculated daily PnL, win rates, trade frequencies, average trade sizes, and drawdown proxies (worst single-trade PnL per day). |
| **Segmentation & Modeling** | Traders grouped into archetypes (e.g., *High-Risk Gambler*, *Consistent Grinder*) using K-Means clustering on risk vs. reward dimensions. A baseline predictive model forecasts next-day profitability. |

---

## 📊 Key Insights

**The Pareto Profitability**
A very small fraction of accounts drive the vast majority of aggregate positive PnL. The overall market relies heavily on high-performing outliers — the average retail trader operates at a steady loss.

**The Win Rate Reality**
The highest concentration of retail traders falls below the 50% success mark. Long-term profitability depends entirely on strict reward-to-risk ratios and minimizing drawdowns, not prediction accuracy.

**Sentiment Drag**
During prolonged Fear regimes, aggregate cumulative PnL flattens or experiences sharp drawdowns. Market panic restricts upside momentum and limits retail edge, causing traders to get chopped out of positions.

---

## 🎯 Actionable Strategy Recommendations

### Rule 1 — Dynamic Position Sizing & Leverage Capping

| | |
|---|---|
| **Trigger** | Market Sentiment Index drops into *Fear* or *Extreme Fear* |
| **Action** | Algorithmically reduce maximum allowable trade sizes and hard-cap leverage at **5×** for active traders |
| **Rationale** | Downside volatility spikes significantly during Fear regimes. Uncapped leverage leads to account-liquidating drawdowns when market structure breaks down, as evidenced by sharp drops in cumulative PnL during these periods. |

### Rule 2 — Restrict Trade Frequency on High-Volatility Days

| | |
|---|---|
| **Trigger** | Sentiment drops to *Fear* **AND** a trader's rolling win rate falls below **45%** |
| **Action** | Reduce the maximum daily trades allowed for that specific user account |
| **Rationale** | Over-trading in choppy, fearful markets without a strict directional edge amplifies losses. Restricting frequency forces selectivity and protects capital in low-probability environments. |

Dashboard:
