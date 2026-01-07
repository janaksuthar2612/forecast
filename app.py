# Updated sales_forecast_app with Groq/LLM explainability & automation integration
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import pickle
import time
import requests  # used for LLM/Groq HTTP calls

# Attempt to import shap (optional). Explainability will fall back to feature_importances_ if not available.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ────────────────────────────────────────────────────────────────
# 1. Shelf-life mapping (days)
# ────────────────────────────────────────────────────────────────
SHELF_LIFE_MAP = {
    'Actimel': 28,
    'Activa Set YoghLight': 21,
    'Activia Greek Drink': 21,
    'Activia Laban FF': 14,
    'Activia Laban Light': 14,
    'Activia Set Yoghurt': 21,
    'Activia Stirred Youg': 21,
    'Activia YGO': 28,
    'Alpro': 180,
    'Cheese': 60,
    'Cream': 14,
    'Creme Caramel': 90,
    'Danao': 180,
    'Danette': 180,
    'Dunkin': 90,
    'Fresh Juice': 5,
    'Greek Yoghurt': 21,
    'Laban': 14,
    'Labneh UHT': 180,
    'Milk': 7,
    'Multi Purpose Cream': 90,
    'Organic Juice': 180,
    'Rashaka Milk': 180,
    'Safio Fresh Drink': 7,
    'Safio Fresh Spoon': 21,
    'Safio UHT': 180,
    'UHT Milk': 180,
    'Yoghurt': 21,
    # Add more brands if needed
}

# ────────────────────────────────────────────────────────────────
# 2. Column renaming mapping
# ────────────────────────────────────────────────────────────────
COLUMN_MAPPING = {
    'Channel Name': 'channel',
    'English Month': 'month',
    'Calendar Year': 'year',
    'Brand': 'brand',
    'Prod Name': 'product',
    'Net Sales Value CAF': 'net_sales_value',
    'Exp Returns Litres': 'exp_returns_litres',
    'Gross Sales Lit': 'gross_sales_lit',
    'Exp Returns Value': 'exp_returns_value'
}

MONTH_TO_NUM = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# ────────────────────────────────────────────────────────────────
# 3. Data loading & preparation function
# ────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns=COLUMN_MAPPING)

    # Required columns check
    required = ['channel', 'month', 'year', 'brand', 'net_sales_value', 'exp_returns_value']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns after renaming: {missing}\nPlease check column names in Excel.")
        st.stop()

    # Data cleaning
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['net_sales_value'] = pd.to_numeric(df['net_sales_value'], errors='coerce').clip(lower=0)
    df['exp_returns_value'] = pd.to_numeric(df['exp_returns_value'], errors='coerce').clip(lower=0)

    # Month number
    df['month_num'] = df['month'].map(MONTH_TO_NUM)
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01',
        errors='coerce'
    )

    # Drop invalid dates
    df = df.dropna(subset=['date'])

    # Add shelf life
    df['shelf_life_days'] = df['brand'].map(SHELF_LIFE_MAP).fillna(30)

    # Monthly aggregation
    monthly = df.groupby(['date', 'brand']).agg({
        'net_sales_value': 'sum',
        'exp_returns_value': 'sum',
        'shelf_life_days': 'first'
    }).reset_index().rename(columns={
        'net_sales_value': 'sales',
        'exp_returns_value': 'returns'
    })

    # Sort
    monthly = monthly.sort_values(['brand', 'date'])

    return monthly, df


# ────────────────────────────────────────────────────────────────
# 4. Add lag features
# ────────────────────────────────────────────────────────────────
def add_lags(df, n_lags=3):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}_sales'] = df.groupby('brand')['sales'].shift(lag)
        df[f'lag_{lag}_returns'] = df.groupby('brand')['returns'].shift(lag)
    return df


# ────────────────────────────────────────────────────────────────
# 5. Train XGBoost model for one brand
# ────────────────────────────────────────────────────────────────
def train_xgboost_brand(product_data, target_col='sales'):
    if len(product_data) < 10:
        return None, None, None, None

    features = ['month_num', 'shelf_life_days']
    for i in range(1, 4):
        features += [f'lag_{i}_sales', f'lag_{i}_returns']

    available_features = [f for f in features if f in product_data.columns]
    X = product_data[available_features]
    y = product_data[target_col]

    if len(X) < 8:
        return None, None, None, None

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=80,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )

    model.fit(X, y)
    return model, available_features, X, y


# Utility: compute simple explainability table (feature importance or SHAP)
def compute_feature_importance(model, feature_names, X_sample=None):
    """
    Returns a DataFrame with feature | importance | method.
    If SHAP is available and X_sample is provided, compute mean(|SHAP|) per feature.
    Otherwise, use model.feature_importances_.
    """
    if SHAP_AVAILABLE and X_sample is not None:
        try:
            # Tree explainer for XGBoost
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X_sample)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap,
                'method': 'shap_mean_abs'
            }).sort_values('importance', ascending=False)
            return fi_df
        except Exception:
            pass

    # Fallback to model importance
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'method': 'xgb_gain'
        }).sort_values('importance', ascending=False)
        return fi_df
    except Exception:
        # final fallback: zero importance
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [0.0] * len(feature_names),
            'method': 'none'
        }).sort_values('importance', ascending=False)
        return fi_df


# Utility: append a small log row to CSV (creates file if not exists)
def append_log_row(filename, row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(filename):
        try:
            df_existing = pd.read_csv(filename)
            df_to_write = pd.concat([df_existing, df], ignore_index=True)
            df_to_write.to_csv(filename, index=False)
        except Exception:
            # fallback to append mode
            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    else:
        df.to_csv(filename, index=False)


# LLM/Groq call helper (generic REST)
def call_llm_groq(prompt: str, api_key: str, endpoint: str, model: str, max_tokens: int = 1024, timeout: int = 20) -> str:
    """
    Generic LLM call helper. The function expects a Bearer-style API key and a JSON REST endpoint.
    Replace/adapt payload and parsing to match your LLM provider (Groq or vendor-specific).
    """
    if not api_key or not endpoint:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": prompt,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Generic parsing: try several common shapes
        if isinstance(data, dict):
            # Common pattern: {"choices": [{"text": "..."}, ...]} or {"output": "..."} or {"data": {"text": "..."}}
            if "choices" in data and isinstance(data["choices"], list) and "text" in data["choices"][0]:
                return data["choices"][0]["text"]
            if "output" in data and isinstance(data["output"], str):
                return data["output"]
            if "text" in data and isinstance(data["text"], str):
                return data["text"]
            # Fallback: return pretty JSON string
            return json.dumps(data, indent=2)
        else:
            return str(data)
    except Exception as e:
        return f"[LLM call failed: {e}]"


# LLM placeholder summary generator (replace with actual LLM call)
def generate_llm_summary(report: dict, groq_conf: dict = None) -> str:
    """
    If groq_conf provided and valid, call the LLM (Groq) to produce an explanation and recommended actions.
    Otherwise fall back to the templated summary.
    """
    if groq_conf and groq_conf.get("api_key") and groq_conf.get("endpoint") and groq_conf.get("model"):
        # Build a compact prompt combining metrics + KPIs + short table of top features
        prompt_obj = {
            "role": "system",
            "content": "You are a senior supply-chain expert. Produce a clear explanation why the forecasting model produced the outputs, prioritized actions to reduce reverse logistics (returns), estimated return-volume and cost savings, and a short automation rule set. Output must be JSON with keys: explanation, actions (list of {priority, action, rationale}), estimates (estimated_returns_avoided_volume, estimated_reverse_logistics_cost_saving, percent_reduction), automation_rules (list). Keep numbers and assumptions explicit."
        }

        features_sales = report.get("feature_importance_sales", [])[:6]
        features_returns = report.get("feature_importance_returns", [])[:6]

        prompt_body = {
            "brand": report["data_provenance"].get("brand"),
            "history_months": report["data_provenance"].get("history_months"),
            "training_start": report["data_provenance"].get("training_start"),
            "training_end": report["data_provenance"].get("training_end"),
            "model_metrics": report.get("training_metrics", {}),
            "top_sales_features": features_sales,
            "top_returns_features": features_returns,
            "kpis": report.get("kpis", {}),
            "notes": report.get("notes", "")
        }

        full_prompt = prompt_obj["content"] + "\n\n" + json.dumps(prompt_body, indent=2)
        llm_text = call_llm_groq(full_prompt, groq_conf['api_key'], groq_conf['endpoint'], groq_conf['model'], max_tokens=800)
        # If LLM returns JSON-like string, try to parse
        try:
            parsed = json.loads(llm_text)
            # Pretty-print if parsed
            return json.dumps(parsed, indent=2)
        except Exception:
            return llm_text

    # Fallback templated summary (existing behavior)
    s = []
    s.append(f"Brand: {report['data_provenance'].get('brand')}")
    s.append(f"Training data: {report['data_provenance'].get('training_start')} to {report['data_provenance'].get('training_end')} ({report['data_provenance'].get('history_months')} months)")
    s.append(f"Sales RMSE (train): {report['training_metrics'].get('sales_rmse')}")
    s.append(f"Returns RMSE (train): {report['training_metrics'].get('returns_rmse')}")
    top_sales = ", ".join([f['feature'] for f in report.get('feature_importance_sales', [])[:3]])
    top_returns = ", ".join([f['feature'] for f in report.get('feature_importance_returns', [])[:3]])
    s.append("Top sales features: " + (top_sales or "N/A"))
    s.append("Top returns features: " + (top_returns or "N/A"))
    s.append("\nSuggested action: Please review forecasts and safety buffer overrides; consider holding manual stock for high-risk SKUs.")
    return "\n".join(s)


# ────────────────────────────────────────────────────────────────
# 6. Main App
# ────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Sales & Returns AI Forecaster", layout="wide")

    st.title("📊 AI Sales Forecasting & Return Reduction Dashboard")
    st.markdown(f"**Current date:** {datetime.now().date()} | Using XGBoost + Shelf-life feature")

    # File uploader for sales input
    uploaded_file = st.sidebar.file_uploader("Upload your sales Excel file", type=["xlsx"])

    # --- New: Groq / LLM configuration in sidebar ---
    st.sidebar.header("LLM / Groq Integration (optional)")
    groq_key_file = st.sidebar.file_uploader("Upload Groq API key (text file)", type=["txt", "key"])
    groq_api_key = None
    if groq_key_file is not None:
        try:
            groq_api_key = groq_key_file.read().decode().strip()
            st.sidebar.success("Groq key loaded (in-memory).")
        except Exception:
            st.sidebar.error("Could not read Groq key file. Ensure it's a plain text file with the API key.")
            groq_api_key = None

    groq_endpoint = st.sidebar.text_input("Groq/LLM endpoint (full URL)", value="")
    groq_model = st.sidebar.text_input("LLM model name (id)", value="")

    # Reverse logistics cost per unit (used to estimate cost saving)
    st.sidebar.header("Cost / Savings inputs")
    reverse_cost_per_unit = st.sidebar.number_input("Estimated reverse logistics cost per unit", min_value=0.0, value=5.0, step=0.1)

    # File required check
    if uploaded_file is None:
        st.info("Please upload your '3 sales.xlsx' file to start.")
        st.stop()

    with st.spinner("Loading and preparing data..."):
        monthly, raw_df = load_and_prepare_data(uploaded_file)

    # ── Sidebar controls ────────────────────────────────────────────
    st.sidebar.header("Forecast Settings")

    available_brands = sorted(monthly['brand'].unique())
    selected_brand = st.sidebar.selectbox("Select Brand", available_brands)

    forecast_months = st.sidebar.slider("Forecast next months", 1, 18, 6)
    lag_count = st.sidebar.slider("Number of lags", 1, 6, 3)

    # Human override for buffer multiplier
    st.sidebar.header("Human Overrides & Explainability")
    human_buffer_override = st.sidebar.number_input(
        "Safety buffer multiplier (human override)",
        min_value=0.5, max_value=3.0, value=1.15, step=0.01,
        help="Override default buffer (default computed from shelf life). Use to simulate more/less conservative stocking."
    )
    show_explain = st.sidebar.checkbox("Show Explainability & Human-AI notes", value=True)

    # Qualitative overrides & uplifts
    st.sidebar.header("Qualitative overrides & Simulations")
    event_note = st.sidebar.text_input("Optional event note (e.g., 'Eid promotion')", value="")
    uplift_percent = st.sidebar.number_input("Event uplift on sales (%)", min_value=-100.0, max_value=500.0, value=0.0, step=0.5)
    uplift_months = st.sidebar.slider("Uplift months (apply to next N months)", min_value=0, max_value=12, value=0)

    # Simulation: returns spike
    returns_spike_pct = st.sidebar.slider("Simulate returns spike (%)", -100, 500, 0, step=5, help="Apply a simulated % change to forecasted returns to see impact.")

    # Escalation threshold
    st.sidebar.header("Automation Controls")
    rmse_escalation_threshold = st.sidebar.number_input("Escalation RMSE threshold (sales)", min_value=0.0, value=1.0, step=0.1)
    auto_approve_low_risk = st.sidebar.checkbox("Auto-approve low-risk recommendations (RMSE below threshold)", value=False)

    # Feedback log user identity (prefill from current user)
    current_user = st.sidebar.text_input("User (for logs)", value="janaksuthar2612")

    # Filter data for selected brand
    product_data = monthly[monthly['brand'] == selected_brand].copy()

    if len(product_data) < 12:
        st.warning(f"Not enough historical data for {selected_brand} (only {len(product_data)} months)")
        st.stop()

    # Add lags
    product_data = add_lags(product_data, lag_count)
    product_data = product_data.dropna()

    product_data['month_num'] = product_data['date'].dt.month

    # ── Train models ────────────────────────────────────────────────
    model_sales, features_sales, X_sales, y_sales = train_xgboost_brand(product_data, 'sales')
    model_returns, features_returns, X_returns, y_returns = train_xgboost_brand(product_data, 'returns')

    if model_sales is None or model_returns is None:
        st.error("Could not train models - too few data points after lagging.")
        st.stop()

    # Compute training RMSE for both models
    try:
        train_pred_sales = model_sales.predict(X_sales)
        train_rmse_sales = mean_squared_error(y_sales, train_pred_sales, squared=False)
    except Exception:
        train_rmse_sales = None

    try:
        train_pred_returns = model_returns.predict(X_returns)
        train_rmse_returns = mean_squared_error(y_returns, train_pred_returns, squared=False)
    except Exception:
        train_rmse_returns = None

    # ── Forecasting ─────────────────────────────────────────────────
    def run_forecast(models_tuple, apply_uplift=True, returns_spike=0.0):
        model_sales, features_sales = models_tuple['sales']
        model_returns, features_returns = models_tuple['returns']
        current_row = product_data.iloc[-1].copy()
        forecast_dates = []
        sales_forecast = []
        returns_forecast = []
        recommended_stock = []
        working_row = current_row.copy()

        for i in range(forecast_months):
            next_date = working_row['date'] + pd.DateOffset(months=1)
            forecast_dates.append(next_date)
            month_num = next_date.month

            # prepare features
            X_next_sales = pd.DataFrame([{
                'month_num': month_num,
                'shelf_life_days': working_row['shelf_life_days'],
                **{f'lag_{j}_sales': working_row.get(f'lag_{j}_sales', 0) for j in range(1, lag_count+1)},
                **{f'lag_{j}_returns': working_row.get(f'lag_{j}_returns', 0) for j in range(1, lag_count+1)}
            }])

            if features_sales:
                for col in features_sales:
                    if col not in X_next_sales.columns:
                        X_next_sales[col] = 0
                X_next_sales = X_next_sales[features_sales]

            pred_sales = max(0, model_sales.predict(X_next_sales)[0])

            # apply qualitative uplift for the first uplift_months if requested
            if apply_uplift and uplift_months > 0 and i < uplift_months and uplift_percent != 0:
                pred_sales = pred_sales * (1.0 + uplift_percent / 100.0)

            sales_forecast.append(pred_sales)

            # returns prediction (use predicted sales as needed)
            X_next_returns = X_next_sales.copy()
            if features_returns and 'lag_1_sales' in features_returns:
                X_next_returns['lag_1_sales'] = pred_sales
            if features_returns:
                for col in features_returns:
                    if col not in X_next_returns.columns:
                        X_next_returns[col] = 0
                X_next_returns = X_next_returns[features_returns]

            pred_returns = max(0, model_returns.predict(X_next_returns)[0])

            # apply simulated returns spike
            if returns_spike != 0:
                pred_returns = pred_returns * (1.0 + returns_spike / 100.0)

            returns_forecast.append(pred_returns)

            default_buffer = 1.15 if working_row['shelf_life_days'] <= 14 else 1.05
            buffer = human_buffer_override if human_buffer_override is not None else default_buffer
            stock = pred_sales - pred_returns * buffer
            recommended_stock.append(max(0, stock))

            # update lags
            for lag in range(lag_count, 1, -1):
                working_row[f'lag_{lag}_sales'] = working_row.get(f'lag_{lag-1}_sales', 0)
                working_row[f'lag_{lag}_returns'] = working_row.get(f'lag_{lag-1}_returns', 0)
            working_row['lag_1_sales'] = pred_sales
            working_row['lag_1_returns'] = pred_returns
            working_row['date'] = next_date

        df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast Sales': np.array(sales_forecast),
            'Forecast Returns': np.array(returns_forecast),
            'Recommended Stock': np.array(recommended_stock)
        })
        return df

    models_tuple = {
        'sales': (model_sales, features_sales),
        'returns': (model_returns, features_returns)
    }

    df_forecast = run_forecast(models_tuple, apply_uplift=True, returns_spike=0.0)

    # Also compute a simulated forecast for returns spike to show delta
    if returns_spike_pct != 0:
        df_sim = run_forecast(models_tuple, apply_uplift=True, returns_spike=returns_spike_pct)
        df_compare = df_forecast.copy()
        df_compare['Sim Recommended Stock'] = df_sim['Recommended Stock']
        df_compare['Delta Stock'] = df_compare['Sim Recommended Stock'] - df_compare['Recommended Stock']
    else:
        df_compare = df_forecast.copy()
        df_compare['Sim Recommended Stock'] = df_forecast['Recommended Stock']
        df_compare['Delta Stock'] = 0.0

    # ── Results Visualization ───────────────────────────────────────
    st.subheader(f"Forecast for **{selected_brand}** ({forecast_months} months)")

    col1, col2 = st.columns(2)

    with col1:
        fig_sales = go.Figure()
        fig_sales.add_trace(go.Scatter(
            x=product_data['date'],
            y=product_data['sales'],
            mode='lines+markers',
            name='Historical Sales'
        ))
        fig_sales.add_trace(go.Scatter(
            x=df_forecast['Date'],
            y=df_forecast['Forecast Sales'],
            mode='lines+markers',
            name='Forecast Sales',
            line=dict(dash='dash', color='orange')
        ))
        fig_sales.update_layout(title="Sales", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_sales, use_container_width=True)

    with col2:
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=product_data['date'],
            y=product_data['returns'],
            mode='lines+markers',
            name='Historical Returns'
        ))
        fig_returns.add_trace(go.Scatter(
            x=df_forecast['Date'],
            y=df_forecast['Forecast Returns'],
            mode='lines+markers',
            name='Forecast Returns',
            line=dict(dash='dash', color='red')
        ))
        fig_returns.update_layout(title="Returns", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_returns, use_container_width=True)

    # Recommendation + interactive approval
    st.subheader("Inventory Recommendation")
    display_df = df_compare.copy()
    display_df_display = display_df.copy()
    display_df_display['Forecast Sales'] = display_df_display['Forecast Sales'].round(0).astype(int)
    display_df_display['Forecast Returns'] = display_df_display['Forecast Returns'].round(0).astype(int)
    display_df_display['Recommended Stock'] = display_df_display['Recommended Stock'].round(0).astype(int)
    display_df_display['Sim Recommended Stock'] = display_df_display['Sim Recommended Stock'].round(0).astype(int)
    display_df_display['Delta Stock'] = display_df_display['Delta Stock'].round(0).astype(int)

    st.dataframe(display_df_display, use_container_width=True)

    total_reduction = np.sum(display_df['Recommended Stock']) - np.sum(display_df['Forecast Sales'])
    st.metric(
        "Estimated Return Reduction (value)",
        f"{total_reduction:,.0f}",
        delta_color="normal" if total_reduction < 0 else "inverse"
    )

    # Compute estimated returns avoided and reverse logistics savings using buffer approach
    total_forecast_returns = df_forecast['Forecast Returns'].sum()
    # Use human_buffer_override as the buffer assumption to compute potential avoided returns volume
    buffer_used = float(human_buffer_override) if human_buffer_override is not None else 1.15
    estimated_returns_avoided_volume = buffer_used * total_forecast_returns
    estimated_reverse_logistics_cost_saving = estimated_returns_avoided_volume * float(reverse_cost_per_unit)
    pct_reduction = (estimated_returns_avoided_volume / total_forecast_returns * 100.0) if total_forecast_returns > 0 else 0.0

    st.markdown("### Estimated supply-chain impact (approx)")
    st.write({
        "total_forecast_returns_volume": float(total_forecast_returns),
        "estimated_returns_avoided_volume": float(np.round(estimated_returns_avoided_volume, 2)),
        "estimated_reverse_logistics_cost_saving": float(np.round(estimated_reverse_logistics_cost_saving, 2)),
        "percent_reduction_estimate": float(np.round(pct_reduction, 2))
    })

    # Auto-approve logic for low-risk
    auto_approved = False
    if auto_approve_low_risk and train_rmse_sales is not None and train_rmse_sales < rmse_escalation_threshold:
        auto_approved = True

    st.markdown("### Human Approval")
    if auto_approved:
        st.success("Auto-approved: model RMSE is below the escalation threshold.")
    else:
        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button("✅ Approve recommendation"):
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "user": current_user,
                    "brand": selected_brand,
                    "action": "approve",
                    "event_note": event_note,
                    "uplift_percent": uplift_percent,
                    "uplift_months": uplift_months,
                    "returns_spike_pct": returns_spike_pct,
                    "forecast_months": forecast_months,
                    "train_rmse_sales": train_rmse_sales,
                    "train_rmse_returns": train_rmse_returns
                }
                append_log_row("decisions_log.csv", row)
                st.success("Decision logged: Approved.")

        with col_reject:
            if st.button("❌ Reject recommendation"):
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "user": current_user,
                    "brand": selected_brand,
                    "action": "reject",
                    "event_note": event_note,
                    "uplift_percent": uplift_percent,
                    "uplift_months": uplift_months,
                    "returns_spike_pct": returns_spike_pct,
                    "forecast_months": forecast_months,
                    "train_rmse_sales": train_rmse_sales,
                    "train_rmse_returns": train_rmse_returns
                }
                append_log_row("decisions_log.csv", row)
                st.warning("Decision logged: Rejected.")

    # Escalation panel
    if train_rmse_sales is not None and train_rmse_sales > rmse_escalation_threshold:
        st.error(f"High model uncertainty: Sales train RMSE = {train_rmse_sales:.2f} (>{rmse_escalation_threshold}). Consider escalation.")
        if st.button("Request escalation / human review"):
            report = {
                "data_provenance": {
                    "brand": selected_brand,
                    "history_months": int(len(product_data)),
                    "training_start": str(product_data['date'].min().date()),
                    "training_end": str(product_data['date'].max().date()),
                    "rows_used_for_training_sales": int(X_sales.shape[0]) if X_sales is not None else 0,
                    "rows_used_for_training_returns": int(X_returns.shape[0]) if X_returns is not None else 0,
                    "model_algorithm": "XGBoostRegressor",
                    "model_params_sales": model_sales.get_params() if model_sales else {},
                    "model_params_returns": model_returns.get_params() if model_returns else {},
                    "shap_available": SHAP_AVAILABLE
                },
                "training_metrics": {
                    "sales_rmse": float(train_rmse_sales) if train_rmse_sales is not None else None,
                    "returns_rmse": float(train_rmse_returns) if train_rmse_returns is not None else None
                },
                "feature_importance_sales": compute_feature_importance(model_sales, features_sales, X_sample=X_sales.head(200)).to_dict(orient='records'),
                "feature_importance_returns": compute_feature_importance(model_returns, features_returns, X_sample=X_returns.head(200)).to_dict(orient='records'),
            }
            groq_conf = {
                "api_key": groq_api_key,
                "endpoint": groq_endpoint,
                "model": groq_model
            } if groq_api_key else None
            summary = generate_llm_summary(report, groq_conf)
            append_log_row("escalations_log.csv", {
                "timestamp": datetime.now().isoformat(),
                "user": current_user,
                "brand": selected_brand,
                "train_rmse_sales": train_rmse_sales,
                "reason": "rmse_above_threshold",
                "llm_summary": summary
            })
            st.info("Escalation requested and logged. Summary:")
            st.code(summary)

    # ── Explainability & Human-AI Interaction Panel ─────────────────
    if show_explain:
        st.header("Explainability & Human–AI Interaction")
        # Data provenance
        data_prov = {
            "brand": selected_brand,
            "history_months": int(len(product_data)),
            "training_start": str(product_data['date'].min().date()),
            "training_end": str(product_data['date'].max().date()),
            "rows_used_for_training_sales": int(X_sales.shape[0]) if X_sales is not None else 0,
            "rows_used_for_training_returns": int(X_returns.shape[0]) if X_returns is not None else 0,
            "model_algorithm": "XGBoostRegressor",
            "model_params_sales": model_sales.get_params() if model_sales else {},
            "model_params_returns": model_returns.get_params() if model_returns else {},
            "shap_available": SHAP_AVAILABLE
        }

        st.subheader("Why this output?")
        st.markdown(
            """
            - Models: We trained separate XGBoost regression models for Sales and Returns using past monthly totals and lag features.
            - Features: month number, shelf-life (days), and up to N lagged sales/returns (set in sidebar).
            - Forecasting: we perform multi-step forecasting by using predicted sales as input for next-step returns.
            - Safety stock: recommended stock = forecast_sales - buffer*forecast_returns. The default buffer depends on shelf-life but you can override it in the sidebar.
            - Explainability: below we show model fit (training RMSE) and per-feature importance. If SHAP is available, we use mean(|SHAP|) as a feature importance proxy.
            """
        )

        # Model performance box
        st.subheader("Model fit (training)")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Sales model RMSE (train)", f"{train_rmse_sales:.2f}" if train_rmse_sales is not None else "N/A")
        with cols[1]:
            st.metric("Returns model RMSE (train)", f"{train_rmse_returns:.2f}" if train_rmse_returns is not None else "N/A")

        # Feature importance
        st.subheader("Feature importance / contribution")
        fi_sales = compute_feature_importance(model_sales, features_sales, X_sample=X_sales.head(200) if X_sales is not None else None)
        fi_returns = compute_feature_importance(model_returns, features_returns, X_sample=X_returns.head(200) if X_returns is not None else None)

        # Display importance tables and bar charts
        st.markdown("Sales model feature importance:")
        st.dataframe(fi_sales.reset_index(drop=True).assign(importance=lambda df: df['importance'].round(5)))

        fig_fi_sales = px.bar(fi_sales, x='importance', y='feature', orientation='h', title='Sales feature importance (higher = more impact)')
        st.plotly_chart(fig_fi_sales, use_container_width=True)

        st.markdown("Returns model feature importance:")
        st.dataframe(fi_returns.reset_index(drop=True).assign(importance=lambda df: df['importance'].round(5)))

        fig_fi_returns = px.bar(fi_returns, x='importance', y='feature', orientation='h', title='Returns feature importance (higher = more impact)', color_discrete_sequence=['salmon'])
        st.plotly_chart(fig_fi_returns, use_container_width=True)

        # Human verification checklist
        st.subheader("Human verification checklist (recommended)")
        st.markdown("""
        1. Confirm the date range and number of historical months used for training. Models require enough historical points — preferably 24+ months for stable seasonality.
        2. Inspect feature importance: if month_num or lag features dominate, models are mainly autoregressive/seasonal.
        3. Check RMSE: compare RMSE to average sales magnitude — is error acceptable?
        4. Validate outliers and data quality in the raw Excel (e.g., mis-entered months, duplicated rows).
        5. Test human overrides: adjust the safety buffer and observe recommended stock changes.
        6. If you suspect bias or unexpected outputs, download the explainability report and review model params and training rows.
        """)

        # Explanation report download
        st.subheader("Download explanation report")
        report = {
            "data_provenance": data_prov,
            "training_metrics": {
                "sales_rmse": float(train_rmse_sales) if train_rmse_sales is not None else None,
                "returns_rmse": float(train_rmse_returns) if train_rmse_returns is not None else None
            },
            "feature_importance_sales": fi_sales[['feature', 'importance', 'method']].to_dict(orient='records'),
            "feature_importance_returns": fi_returns[['feature', 'importance', 'method']].to_dict(orient='records'),
            "human_buffer_override": float(human_buffer_override),
            "kpis": {
                "total_forecast_returns": float(total_forecast_returns),
                "estimated_returns_avoided_volume": float(np.round(estimated_returns_avoided_volume, 2)),
                "estimated_reverse_logistics_cost_saving": float(np.round(estimated_reverse_logistics_cost_saving, 2)),
                "reverse_cost_per_unit": float(reverse_cost_per_unit)
            }
        }
        report_text = json.dumps(report, indent=2)
        st.download_button("📥 Download Explainability JSON", report_text, file_name=f"{selected_brand}_explainability_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")

        # --- New: Request Groq expert recommendations ---
        st.subheader("LLM-driven supply-chain recommendations (Groq)")

        groq_conf = {
            "api_key": groq_api_key,
            "endpoint": groq_endpoint,
            "model": groq_model
        } if groq_api_key and groq_endpoint and groq_model else None

        st.markdown("Use the Groq/LLM to produce an expert explanation & prioritized actions. Provide Groq key + endpoint & model in the sidebar to enable.")
        if st.button("Get Groq Expert Recommendation"):
            with st.spinner("Generating expert recommendation from LLM..."):
                # Build report similar to escalation
                lreport = {
                    "data_provenance": report["data_provenance"],
                    "training_metrics": report["training_metrics"],
                    "feature_importance_sales": report["feature_importance_sales"],
                    "feature_importance_returns": report["feature_importance_returns"],
                    "kpis": report["kpis"],
                    "notes": {
                        "event_note": event_note,
                        "uplift_percent": uplift_percent,
                        "uplift_months": uplift_months,
                        "returns_spike_pct": returns_spike_pct,
                        "buffer_used": buffer_used
                    }
                }
                summary = generate_llm_summary(lreport, groq_conf)
                append_log_row("llm_recommendations_log.csv", {
                    "timestamp": datetime.now().isoformat(),
                    "user": current_user,
                    "brand": selected_brand,
                    "kpis": json.dumps(lreport.get("kpis", {})),
                    "llm_output": summary
                })
                st.info("LLM recommendation (raw):")
                st.code(summary)

                # If the LLM returned JSON, display parsed actions in a structured way
                try:
                    parsed = json.loads(summary)
                    if isinstance(parsed, dict):
                        st.markdown("**LLM Structured Output**")
                        if "explanation" in parsed:
                            st.markdown("**Explanation**")
                            st.write(parsed["explanation"])
                        if "actions" in parsed:
                            st.markdown("**Actions (prioritized)**")
                            for a in parsed["actions"]:
                                st.write(f"- Priority {a.get('priority', '?')}: {a.get('action')} — {a.get('rationale','')}")
                        if "estimates" in parsed:
                            st.markdown("**Estimates**")
                            st.write(parsed["estimates"])
                        if "automation_rules" in parsed:
                            st.markdown("**Suggested automation rules**")
                            for r in parsed["automation_rules"]:
                                st.write(f"- {r}")
                except Exception:
                    # Not JSON — treat as plain text
                    pass

                # Show an "Apply suggested automation" button (logs the intent)
                if st.button("Apply / Log suggested automation (no external effect)"):
                    append_log_row("automation_log.csv", {
                        "timestamp": datetime.now().isoformat(),
                        "user": current_user,
                        "brand": selected_brand,
                        "llm_output": summary,
                        "action_taken": "logged_apply"
                    })
                    st.success("Suggested automation logged. (This action is only logged; integrate with your operational systems to enact.)")

    # ── Feedback Loop: upload actuals & rate the explanation ───────────
    st.header("Post-decision feedback & retraining")
    st.markdown("Upload actuals (Date, Actual Sales, Actual Returns) to record outcomes and compute error vs forecast.")
    actuals_file = st.file_uploader("Upload actuals CSV", type=["csv"], key="actuals_uploader")

    if actuals_file is not None:
        try:
            actuals = pd.read_csv(actuals_file, parse_dates=['Date'])
            # merge with forecast on Date
            merged = pd.merge(df_forecast, actuals, on='Date', how='inner')
            if merged.empty:
                st.warning("No matching dates between forecast and uploaded actuals.")
            else:
                merged['sales_error'] = merged['Actual Sales'] - merged['Forecast Sales']
                merged['returns_error'] = merged['Actual Returns'] - merged['Forecast Returns']
                merged_summary = {
                    "timestamp": datetime.now().isoformat(),
                    "user": current_user,
                    "brand": selected_brand,
                    "rows": len(merged),
                    "mean_sales_error": float(merged['sales_error'].mean()),
                    "mean_returns_error": float(merged['returns_error'].mean())
                }
                append_log_row("feedback_log.csv", merged_summary)
                st.success("Actuals processed and feedback logged.")
                st.dataframe(merged.assign(sales_error=lambda d: d['sales_error'].round(2), returns_error=lambda d: d['returns_error'].round(2)))
        except Exception as e:
            st.error(f"Error processing actuals: {e}")

    st.markdown("Rate the explanation & decision (1 = poor, 5 = excellent)")
    rating = st.slider("Explanation clarity rating", 1, 5, 4, step=1)
    comments = st.text_area("Optional comments (what was helpful / unclear?)")

    if st.button("Submit feedback"):
        feedback_row = {
            "timestamp": datetime.now().isoformat(),
            "user": current_user,
            "brand": selected_brand,
            "rating": int(rating),
            "comments": comments,
            "event_note": event_note,
            "uplift_percent": uplift_percent,
            "uplift_months": uplift_months,
            "returns_spike_pct": returns_spike_pct
        }
        append_log_row("feedback_log.csv", feedback_row)
        st.success("Thanks — feedback recorded. This will be used for retraining / continuous improvement.")

    # Retrain button (in-session)
    if st.button("Retrain models now"):
        with st.spinner("Retraining models..."):
            model_sales_new, features_sales_new, X_sales_new, y_sales_new = train_xgboost_brand(product_data, 'sales')
            model_returns_new, features_returns_new, X_returns_new, y_returns_new = train_xgboost_brand(product_data, 'returns')
            if model_sales_new is not None and model_returns_new is not None:
                # overwrite models used for forecasting in-session
                model_sales = model_sales_new
                model_returns = model_returns_new
                features_sales = features_sales_new
                features_returns = features_returns_new
                X_sales = X_sales_new
                X_returns = X_returns_new
                train_pred_sales = model_sales.predict(X_sales)
                train_rmse_sales = mean_squared_error(y_sales, train_pred_sales, squared=False)
                train_pred_returns = model_returns.predict(X_returns)
                train_rmse_returns = mean_squared_error(y_returns, train_pred_returns, squared=False)
                # recompute forecast
                df_forecast = run_forecast({'sales': (model_sales, features_sales), 'returns': (model_returns, features_returns)}, apply_uplift=True, returns_spike=returns_spike_pct)
                st.success("Retraining complete and forecasts updated.")
            else:
                st.error("Retraining failed due to insufficient data or other error.")

    # Download forecast CSV
    csv = df_forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Forecast CSV",
        csv,
        f"{selected_brand}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

    # Provide quick onboarding & tutorials
    with st.expander("How shelf_life_days affects returns (brief tutorial)"):
        st.markdown("""
        - shelf_life_days is included as a feature in the model: shorter shelf life often increases returns (waste) risk and may reduce recommended stock safety.
        - The safety buffer uses shelf-life to be more conservative for perishable items.
        - Try changing the buffer multiplier in the sidebar to see how recommended stock responds.
        - Use the 'Event uplift' to simulate promotions/holidays that temporarily increase demand.
        """)

    with st.expander("How to connect an LLM for summaries and escalation"):
        st.markdown("""
        - Provide Groq API key (text file) and fill endpoint + model name in the sidebar.
        - The app will construct a compact JSON report and ask the LLM (acting as supply-chain expert) for:
          * explanation of forecast,
          * prioritized actions to reduce returns,
          * estimated return-volume avoided and cost savings,
          * suggested automation rules for decisioning.
        - Example (pseudo):
          1) upload key file
          2) set endpoint (e.g., https://api.groq.ai/v1/llm/predict)
          3) set model id/name
          4) click 'Get Groq Expert Recommendation'
        - For production, move keys to a secure secret store & call LLM from a trusted server or backend.
        """)

if __name__ == "__main__":
    main()
