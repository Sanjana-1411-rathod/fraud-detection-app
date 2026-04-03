import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.decomposition import PCA
import joblib
import io

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Space+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f1a2e 50%, #0a0a1a 100%);
}
.metric-card {
    background: linear-gradient(135deg, rgba(0,255,136,0.05) 0%, rgba(0,200,255,0.05) 100%);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.fraud-alert {
    background: linear-gradient(135deg, rgba(255,50,50,0.15), rgba(255,100,0,0.1));
    border: 2px solid rgba(255,50,50,0.5);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.safe-alert {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,200,255,0.1));
    border: 2px solid rgba(0,255,136,0.5);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
h1 { color: #00ff88 !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 700; }
h2, h3 { color: #00c8ff !important; font-family: 'Rajdhani', sans-serif !important; }
.stMetric label { color: #888 !important; font-size: 13px; }
.stMetric [data-testid="stMetricValue"] { color: #00ff88 !important; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────
@st.cache_data
def generate_data(n_samples=5000, fraud_ratio=0.05):
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        'amount': np.random.lognormal(4, 1.2, n_legit),
        'hour': np.random.choice(range(8, 22), n_legit),
        'day_of_week': np.random.randint(0, 7, n_legit),
        'merchant_category': np.random.choice(['retail','food','travel','entertainment','utilities'], n_legit, p=[0.35,0.30,0.10,0.15,0.10]),
        'distance_from_home_km': np.random.exponential(15, n_legit),
        'transaction_count_last_24h': np.random.poisson(3, n_legit),
        'avg_amount_last_30d': np.random.lognormal(4, 0.8, n_legit),
        'account_age_days': np.random.randint(180, 3650, n_legit),
        'failed_attempts': np.random.choice([0, 1], n_legit, p=[0.95, 0.05]),
        'international': np.random.choice([0, 1], n_legit, p=[0.92, 0.08]),
        'is_fraud': 0
    })

    # Fraudulent transactions
    fraud = pd.DataFrame({
        'amount': np.random.lognormal(6, 1.5, n_fraud),
        'hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)), n_fraud),
        'day_of_week': np.random.randint(0, 7, n_fraud),
        'merchant_category': np.random.choice(['retail','food','travel','entertainment','utilities'], n_fraud, p=[0.15,0.10,0.40,0.30,0.05]),
        'distance_from_home_km': np.random.exponential(200, n_fraud),
        'transaction_count_last_24h': np.random.poisson(12, n_fraud),
        'avg_amount_last_30d': np.random.lognormal(4, 0.8, n_fraud),
        'account_age_days': np.random.randint(1, 365, n_fraud),
        'failed_attempts': np.random.choice([0, 1, 2], n_fraud, p=[0.50, 0.30, 0.20]),
        'international': np.random.choice([0, 1], n_fraud, p=[0.40, 0.60]),
        'is_fraud': 1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    df['merchant_category_enc'] = le.fit_transform(df['merchant_category'])
    features = ['amount', 'hour', 'day_of_week', 'merchant_category_enc',
                'distance_from_home_km', 'transaction_count_last_24h',
                'avg_amount_last_30d', 'account_age_days',
                'failed_attempts', 'international']
    X = df[features]
    y = df['is_fraud']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, features, scaler, le

# ─────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────
@st.cache_resource
def train_models(n_samples, fraud_ratio):
    df = generate_data(n_samples, fraud_ratio)
    X_scaled, y, features, scaler, le = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'cm': confusion_matrix(y_test, y_pred)
        }
        trained_models[name] = model

    # Isolation Forest (unsupervised)
    iso = IsolationForest(contamination=fraud_ratio, random_state=42)
    iso.fit(X_train)
    iso_pred = iso.predict(X_test)
    iso_pred_bin = (iso_pred == -1).astype(int)
    results["Isolation Forest"] = {
        'accuracy': accuracy_score(y_test, iso_pred_bin),
        'precision': precision_score(y_test, iso_pred_bin, zero_division=0),
        'recall': recall_score(y_test, iso_pred_bin, zero_division=0),
        'f1': f1_score(y_test, iso_pred_bin, zero_division=0),
        'roc_auc': 0.5,
        'y_test': y_test,
        'y_pred': iso_pred_bin,
        'y_prob': iso_pred_bin.astype(float),
        'cm': confusion_matrix(y_test, iso_pred_bin)
    }
    trained_models["Isolation Forest"] = iso

    return results, trained_models, features, scaler, le, df, X_test, y_test, X_scaled, y

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown("## 🛡️ FraudShield AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🤖 Model Training & Evaluation",
    "🔍 Live Transaction Check",
    "📂 Batch CSV Detection",
    "📈 Feature Analysis"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
n_samples = st.sidebar.slider("Dataset Size", 1000, 10000, 5000, 500)
fraud_ratio = st.sidebar.slider("Fraud Ratio (%)", 1, 20, 5) / 100

# Load data & models
with st.spinner("🔄 Training ML models..."):
    results, trained_models, features, scaler, le, df, X_test, y_test, X_scaled, y = train_models(n_samples, fraud_ratio)

best_model_name = max(results, key=lambda k: results[k]['f1'] if k != "Isolation Forest" else 0)
best_model = trained_models[best_model_name]

# ─────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("🛡️ FraudShield AI — Smart Financial Fraud Detection")
    st.markdown("**Real-time ML-powered fraud intelligence dashboard**")
    st.markdown("---")

    fraud_count = df['is_fraud'].sum()
    total = len(df)
    fraud_pct = fraud_count / total * 100
    avg_fraud_amt = df[df['is_fraud'] == 1]['amount'].mean()
    avg_legit_amt = df[df['is_fraud'] == 0]['amount'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Cases", f"{fraud_count:,}", f"{fraud_pct:.1f}%")
    col3.metric("Avg Fraud Amount", f"₹{avg_fraud_amt:,.0f}")
    col4.metric("Avg Legit Amount", f"₹{avg_legit_amt:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(df, x='amount', color='is_fraud',
                           color_discrete_map={0: '#00ff88', 1: '#ff4444'},
                           labels={'is_fraud': 'Fraud', 'amount': 'Amount (₹)'},
                           nbins=60, barmode='overlay', opacity=0.7,
                           template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud by Hour of Day")
        hourly = df.groupby(['hour', 'is_fraud']).size().reset_index(name='count')
        fig2 = px.bar(hourly, x='hour', y='count', color='is_fraud',
                      color_discrete_map={0: '#00c8ff', 1: '#ff4444'},
                      barmode='group', template='plotly_dark')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fraud by Merchant Category")
        cat_fraud = df.groupby(['merchant_category', 'is_fraud']).size().reset_index(name='count')
        fig3 = px.bar(cat_fraud, x='merchant_category', y='count', color='is_fraud',
                      color_discrete_map={0: '#00ff88', 1: '#ff4444'},
                      template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("International vs Domestic Fraud")
        intl = df.groupby(['international', 'is_fraud']).size().reset_index(name='count')
        intl['international'] = intl['international'].map({0: 'Domestic', 1: 'International'})
        fig4 = px.pie(intl[intl['is_fraud'] == 1], names='international', values='count',
                      color_discrete_sequence=['#ff4444', '#ff8800'],
                      template='plotly_dark')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────
# PAGE: MODEL TRAINING
# ─────────────────────────────────────────
elif page == "🤖 Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    st.subheader("Model Performance Comparison")
    metrics_df = pd.DataFrame({
        name: {
            'Accuracy': round(r['accuracy'], 4),
            'Precision': round(r['precision'], 4),
            'Recall': round(r['recall'], 4),
            'F1 Score': round(r['f1'], 4),
            'ROC-AUC': round(r['roc_auc'], 4) if r['roc_auc'] else 'N/A'
        }
        for name, r in results.items()
    }).T.reset_index().rename(columns={'index': 'Model'})

    st.dataframe(metrics_df.style.highlight_max(subset=['Accuracy','Precision','Recall','F1 Score'],
                 color='#003322'), use_container_width=True)

    # Bar chart comparison
    melt = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    melt['Score'] = pd.to_numeric(melt['Score'], errors='coerce')
    fig = px.bar(melt, x='Model', y='Score', color='Metric', barmode='group',
                 template='plotly_dark', title='Model Metrics Comparison')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # ROC curves
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    colors = ['#00ff88', '#00c8ff', '#ff8800', '#ff4444']
    for (name, r), color in zip(results.items(), colors):
        if name != "Isolation Forest":
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={r['roc_auc']:.3f})",
                                         line=dict(color=color, width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='gray')))
    fig_roc.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrices")
    sel_model = st.selectbox("Select Model", list(results.keys()))
    cm = results[sel_model]['cm']
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Teal',
                       labels=dict(x='Predicted', y='Actual'),
                       x=['Legit', 'Fraud'], y=['Legit', 'Fraud'],
                       template='plotly_dark', title=f'{sel_model} — Confusion Matrix')
    fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cm, use_container_width=True)

# ─────────────────────────────────────────
# PAGE: LIVE TRANSACTION CHECK
# ─────────────────────────────────────────
elif page == "🔍 Live Transaction Check":
    st.title("🔍 Live Transaction Fraud Check")
    st.markdown("Enter transaction details to check for fraud in real-time.")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=500.0, step=50.0)
        hour = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        day_enc = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(day_of_week)
    with col2:
        merchant_category = st.selectbox("Merchant Category", ['retail','food','travel','entertainment','utilities'])
        distance = st.number_input("Distance from Home (km)", min_value=0.0, value=10.0, step=1.0)
        tx_count_24h = st.number_input("Transactions in Last 24h", min_value=0, value=2, step=1)
    with col3:
        avg_amount_30d = st.number_input("Avg Amount Last 30d (₹)", min_value=1.0, value=300.0, step=50.0)
        account_age = st.number_input("Account Age (days)", min_value=1, value=365, step=10)
        failed_attempts = st.slider("Failed Attempts", 0, 5, 0)
        international = st.checkbox("International Transaction")

    model_choice = st.selectbox("Choose Model", [k for k in trained_models if k != "Isolation Forest"])

    if st.button("🔍 Analyze Transaction", use_container_width=True):
        cat_enc = le.transform([merchant_category])[0]
        input_data = np.array([[amount, hour, day_enc, cat_enc, distance,
                                 tx_count_24h, avg_amount_30d, account_age,
                                 failed_attempts, int(international)]])
        input_scaled = scaler.transform(input_data)

        model = trained_models[model_choice]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h1 style="color:#ff4444">⚠️ FRAUD DETECTED</h1>
                    <h2 style="color:#ff8800">Risk Score: {probability:.1%}</h2>
                    <p style="color:#ccc">This transaction has been flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    <h1 style="color:#00ff88">✅ TRANSACTION SAFE</h1>
                    <h2 style="color:#00c8ff">Risk Score: {probability:.1%}</h2>
                    <p style="color:#ccc">This transaction appears legitimate.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Fraud Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4444" if probability > 0.5 else "#00ff88"},
                    'steps': [
                        {'range': [0, 30], 'color': '#003322'},
                        {'range': [30, 60], 'color': '#332200'},
                        {'range': [60, 100], 'color': '#330000'}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 3}, 'value': 50}
                }
            ))
            gauge.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                height=280, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

# ─────────────────────────────────────────
# PAGE: BATCH CSV
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# PAGE: BATCH CSV
# ─────────────────────────────────────────
elif page == "📂 Batch CSV Detection":
    st.title("📂 Batch CSV Fraud Detection")
    st.markdown("Upload your CSV file for fraud detection.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    model_choice = st.selectbox("Model", [k for k in trained_models if k != "Isolation Forest"])

    if uploaded:
        try:
            udf = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(udf)} rows")

            # Preprocess
            udf['merchant_category_enc'] = le.transform(udf['merchant_category'])

            feat_cols = ['amount', 'hour', 'day_of_week', 'merchant_category_enc',
                        'distance_from_home_km', 'transaction_count_last_24h',
                        'avg_amount_last_30d', 'account_age_days',
                        'failed_attempts', 'international']

            X_up = scaler.transform(udf[feat_cols])

            model = trained_models[model_choice]
            preds = model.predict(X_up)
            probs = model.predict_proba(X_up)[:, 1]

            # Add results
            udf['fraud_prediction'] = preds
            udf['fraud_probability'] = probs.round(4)

            # Metrics
            fraud_flagged = udf[udf['fraud_prediction'] == 1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(udf))
            col2.metric("Fraud", len(fraud_flagged))
            col3.metric("Fraud %", f"{len(fraud_flagged)/len(udf)*100:.1f}%")

            # 📊 Graph
            st.subheader("Fraud Probability Distribution")
            fig = px.histogram(
                udf,
                x="fraud_probability",
                nbins=50,
                template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # 📋 Table
            st.subheader("Results Preview")
            st.dataframe(
                udf[['amount', 'merchant_category', 'fraud_prediction', 'fraud_probability']],
                use_container_width=True
            )

            # ⬇️ Download AFTER graph
            out = io.StringIO()
            udf.to_csv(out, index=False)

            st.download_button(
                "⬇️ Download Results CSV",
                out.getvalue(),
                "fraud_results.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────
# PAGE: FEATURE ANALYSIS
# ─────────────────────────────────────────
elif page == "📈 Feature Analysis":
    st.title("📈 Feature Importance & Analysis")

    rf = trained_models["Random Forest"]
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)

    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='teal',
                 template='plotly_dark', title='Random Forest Feature Importances')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA — 2D Visualization of Transactions")
    pca = PCA(n_components=2, random_state=42)
    sample_idx = np.random.choice(len(X_scaled), min(2000, len(X_scaled)), replace=False)
    X_pca = pca.fit_transform(X_scaled[sample_idx])
    y_sample = y.values[sample_idx]
    pca_df = pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1], 'Fraud': y_sample.astype(str)})

    fig2 = px.scatter(pca_df, x='PC1', y='PC2', color='Fraud',
                      color_discrete_map={'0': "#5eba8f", '1': "#d67474"},
                      opacity=0.6, template='plotly_dark',
                      title='PCA — Fraud vs Legit Transactions')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df[['amount', 'hour', 'distance_from_home_km',
                     'transaction_count_last_24h', 'account_age_days', 'is_fraud']]
    corr = numeric_df.corr()
    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu',
                     template='plotly_dark', title='Feature Correlation Matrix')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)
