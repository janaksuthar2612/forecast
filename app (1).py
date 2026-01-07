# File name: sales_forecast_app.py
# Run with:   streamlit run sales_forecast_app.py

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

# Attempt to import shap (optional). Explainability will fall back to feature_importances_ if not available.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ────────────────────────────────────────────────────────────────
# 1. Shelf-life mapping (days) - typical values for dairy/juice products
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
# 2. Column renaming mapping (very important!)
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
# Note: returns training X and y so we can compute metrics & explainability
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
            # shap_values may be 2D (n_samples, n_features)
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


# ────────────────────────────────────────────────────────────────
# 6. Main App
# ────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Sales & Returns AI Forecaster", layout="wide")

    st.title("📊 AI Sales Forecasting & Return Reduction Dashboard")
    st.markdown("**Current date:** December 31, 2025 | Using XGBoost + Shelf-life feature")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your sales Excel file", type=["xlsx"])

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

    # Human override for buffer multiplier (so humans can test alternative safety margins)
    st.sidebar.header("Human Overrides & Explainability")
    human_buffer_override = st.sidebar.number_input(
        "Safety buffer multiplier (human override)",
        min_value=0.5, max_value=3.0, value=1.15, step=0.01,
        help="Override default buffer (default computed from shelf life). Use to simulate more/less conservative stocking."
    )
    show_explain = st.sidebar.checkbox("Show Explainability & Human-AI notes", value=True)

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
    current_row = product_data.iloc[-1].copy()
    forecast_dates = []
    sales_forecast = []
    returns_forecast = []
    recommended_stock = []

    # Working copy of lags inside loop for multi-step forecasting
    working_row = current_row.copy()

    for i in range(forecast_months):
        next_date = working_row['date'] + pd.DateOffset(months=1)
        forecast_dates.append(next_date)

        month_num = next_date.month

        # Prepare features for next month
        X_next_sales = pd.DataFrame([{
            'month_num': month_num,
            'shelf_life_days': working_row['shelf_life_days'],
            **{f'lag_{j}_sales': working_row.get(f'lag_{j}_sales', 0) for j in range(1, lag_count+1)},
            **{f'lag_{j}_returns': working_row.get(f'lag_{j}_returns', 0) for j in range(1, lag_count+1)}
        }])

        # Ensure columns match training feature order to avoid XGBoost feature_names mismatch
        if features_sales:
            # add any missing columns with 0
            for col in features_sales:
                if col not in X_next_sales.columns:
                    X_next_sales[col] = 0
            # reorder to the exact order used at training
            X_next_sales = X_next_sales[features_sales]

        pred_sales = max(0, model_sales.predict(X_next_sales)[0])
        sales_forecast.append(pred_sales)

        # Use predicted sales as feature for returns
        X_next_returns = X_next_sales.copy()
        if features_returns and 'lag_1_sales' in features_returns:
            X_next_returns['lag_1_sales'] = pred_sales

        if features_returns:
            for col in features_returns:
                if col not in X_next_returns.columns:
                    X_next_returns[col] = 0
            X_next_returns = X_next_returns[features_returns]

        pred_returns = max(0, model_returns.predict(X_next_returns)[0])
        returns_forecast.append(pred_returns)

        # Simple safety stock logic (use human override multiplier when provided)
        default_buffer = 1.15 if working_row['shelf_life_days'] <= 14 else 1.05
        buffer = human_buffer_override if human_buffer_override is not None else default_buffer
        stock = pred_sales - pred_returns * buffer
        recommended_stock.append(max(0, stock))

        # Update working_row lags for next iteration
        # shift sales lags down and insert pred_sales as lag_1
        for lag in range(lag_count, 1, -1):
            working_row[f'lag_{lag}_sales'] = working_row.get(f'lag_{lag-1}_sales', 0)
            working_row[f'lag_{lag}_returns'] = working_row.get(f'lag_{lag-1}_returns', 0)
        working_row['lag_1_sales'] = pred_sales
        working_row['lag_1_returns'] = pred_returns
        working_row['date'] = next_date

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
            x=forecast_dates,
            y=sales_forecast,
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
            x=forecast_dates,
            y=returns_forecast,
            mode='lines+markers',
            name='Forecast Returns',
            line=dict(dash='dash', color='red')
        ))
        fig_returns.update_layout(title="Returns", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_returns, use_container_width=True)

    # Recommendation
    st.subheader("Inventory Recommendation")
    df_forecast = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast Sales': np.round(sales_forecast, 0),
        'Forecast Returns': np.round(returns_forecast, 0),
        'Recommended Stock': np.round(recommended_stock, 0)
    })

    st.dataframe(df_forecast.style.format({
        'Forecast Sales': '{:,.0f}',
        'Forecast Returns': '{:,.0f}',
        'Recommended Stock': '{:,.0f}'
    }), use_container_width=True)

    total_reduction = np.sum(recommended_stock) - np.sum(sales_forecast)
    st.metric(
        "Estimated Return Reduction (value)",
        f"{total_reduction:,.0f}",
        delta_color="normal" if total_reduction < 0 else "inverse"
    )

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
            "human_buffer_override": float(human_buffer_override)
        }
        report_text = json.dumps(report, indent=2)
        st.download_button("📥 Download Explainability JSON", report_text, file_name=f"{selected_brand}_explainability_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")

    # Download forecast CSV
    csv = df_forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Forecast CSV",
        csv,
        f"{selected_brand}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()
