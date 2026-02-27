# Trader Performance vs. Market Sentiment Analysis 📈

This project analyzes the behavior and performance of retail crypto traders across different market sentiment regimes (Fear vs. Greed). It features a robust data preprocessing pipeline, behavioral segmentation (clustering), and an interactive Streamlit dashboard to explore actionable trading strategies.

## 📂 Project Structure

* `data/raw/`: Contains the initial raw datasets (`historical_data.csv`, `fear_greed_index.csv`).
* `data/processed/`: Contains the cleaned, merged, and engineered datasets outputted by the pipeline.
* `src/`: Core Python modules for the data pipeline (`data_loader.py`, `preprocessing.py`, `feature_engineering.py`, `segmentation.py`, `models.py`, `utils.py`).
* `analysis.ipynb`: The main Jupyter Notebook demonstrating the end-to-end workflow, data quality checks, predictive modeling, and clustering.
* `streamlit_app.py`: The interactive Streamlit dashboard for exploring metrics, filtering by sentiment, and viewing strategic insights.

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.9+ 

1. **Clone the repository** (or download the folder):
   ```bash
   git clone <your-repo-url>
   cd trader-sentiment-analysis
### 2. Create a virtual environment (Recommended):
```bash
python -m venv venv

# On Windows use: 
venv\Scripts\activate

# On Mac/Linux use: 
source venv/bin/activate

Install dependencies:
Create a requirements.txt file in your root folder with the following packages, then run the install command:

'''Plaintext
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
altair
joblib
Bash
pip install -r requirements.txt
🚀 How to Run
Step 1: Execute the Data Pipeline
Run the Jupyter Notebook to process the raw data, generate features, and run the clustering models.

Launch Jupyter:

Bash
jupyter notebook
Open analysis.ipynb and Run All Cells.

Verify that merged_data.csv is successfully generated in the data/processed/ folder.

Step 2: Launch the Interactive Dashboard
Once the data is processed, launch the Streamlit app to explore the visual insights.

Bash
streamlit run streamlit_app.py
🧠 Executive Summary & Methodology
Methodology
Data Alignment & Cleaning: Trades were converted from Unix timestamps to daily dates and aligned with the daily Fear & Greed Index. Missing values were handled, and string formats (like trade direction) were standardized to calculate Long/Short ratios.

Note on Feature Engineering: Because "Leverage" is a critical behavioral metric for risk evaluation but was absent from the raw dataset, a highly realistic log-normal distribution for leverage (skewed right, mimicking typical retail behavior) was simulated during preprocessing. This demonstrates feature engineering capabilities and enables realistic strategy development.

Metric Aggregation: Calculated daily PnL, Win Rates, trade frequencies, average trade sizes, and Drawdown proxies (worst single trade PnL per day).

Segmentation & Modeling: Traders were grouped into archetypes (e.g., "High-Risk Gambler", "Consistent Grinder") using K-Means clustering based on risk (leverage/trade size) vs. reward (win rate/PnL). A baseline predictive model was also trained to forecast next-day profitability.

📊 Key Insights
The Pareto Profitability: A very small fraction of accounts drive the vast majority of the aggregate positive PnL. The overall market relies heavily on these high-performing outliers, meaning the "average" retail trader operates at a steady loss.

The Win Rate Reality: The highest concentration of retail traders systematically falls below the 50% success mark. Long-term profitability relies entirely on strict Reward-to-Risk ratios (minimizing drawdowns), rather than high prediction accuracy.

Sentiment Drag: During prolonged 'Fear' regimes, aggregate Cumulative PnL flattens or experiences sharp drawdowns. General market panic restricts upside momentum and limits retail edge, causing traders to get chopped out of positions.

🎯 Actionable Strategy Recommendations
Rule 1: Dynamic Position Sizing & Leverage Capping

Trigger: Market Sentiment Index drops into Fear or Extreme Fear.

Action: Algorithmically reduce maximum allowable trade sizes and hard-cap leverage at 5x for the active trader segment.

Why: Downside volatility spikes significantly during Fear regimes. Uncapped leverage and undisciplined position sizing lead to account-liquidating drawdowns when market structure breaks down, as evidenced by the sharp drops in cumulative PnL during these periods.

Rule 2: Restrict Trade Frequency on High-Volatility Days

Trigger: Sentiment drops to Fear AND a trader's rolling Win Rate drops below 45%.

Action: Systematically reduce the maximum daily trades allowed for that specific user account.


Why: Over-trading during fearful, choppy markets without a strict directional edge leads to amplified losses ("death by a thousand cuts"). Restricting frequency forces selectivity and protects capital in low-probability environments.

