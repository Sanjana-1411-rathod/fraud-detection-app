# 🛡️ FraudShield AI — Smart Financial Fraud Detection

## Features
- **4 ML Models**: Random Forest, Gradient Boosting, Logistic Regression, Isolation Forest
- **Live Transaction Check**: Enter any transaction and get instant fraud prediction with risk gauge
- **Batch CSV Detection**: Upload CSVs of transactions for bulk screening
- **Interactive Dashboard**: Charts on fraud patterns by hour, category, amount
- **Model Evaluation**: ROC curves, confusion matrices, F1/AUC comparison
- **Feature Analysis**: Importance plots, PCA visualization, correlation heatmap

## Setup & Run (3 steps)

### Step 1 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Open in browser
The app opens automatically at: **http://localhost:8501**

## Pages
| Page | Description |
|------|-------------|
| 📊 Dashboard | Transaction overview, fraud patterns by time/category |
| 🤖 Model Training & Evaluation | Compare 4 ML models, ROC curves, confusion matrix |
| 🔍 Live Transaction Check | Real-time single transaction fraud prediction |
| 📂 Batch CSV Detection | Upload and screen many transactions at once |
| 📈 Feature Analysis | Feature importance, PCA, correlation heatmap |

## CSV Format (for Batch Detection)
Your CSV must have these columns:
```
amount, hour, day_of_week, merchant_category, distance_from_home_km,
transaction_count_last_24h, avg_amount_last_30d, account_age_days,
failed_attempts, international
```
A sample CSV can be downloaded from inside the app.

## ML Algorithms Used
- **Random Forest** — ensemble of decision trees, best for tabular fraud data
- **Gradient Boosting** — sequential boosting for high accuracy
- **Logistic Regression** — fast baseline, interpretable
- **Isolation Forest** — unsupervised anomaly detection (no labels needed)
