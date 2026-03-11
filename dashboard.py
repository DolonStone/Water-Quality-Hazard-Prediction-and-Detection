"""
Water Quality Anomaly Detection Dashboard

A Streamlit dashboard that:
1. Trains a random forest anomaly detector on historical data (2022-2024)
2. Displays water quality data from the past 24 hours
3. Identifies and visualizes anomalies using the trained model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.anomaly_detection import WaterQualityAnomalyDetector
from src.data import (
    format_data_for_modeling,
    get_instantaneous_data,
    get_historical_data,
    MONITORING_STATIONS,
    WATER_QUALITY_PARAMS,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Water Quality Anomaly Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { max-width: 1400px; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .anomaly-true {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .anomaly-false {
        background-color: #ccffcc;
        color: #00cc00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def init_session():
    """Initialize session state and model."""
    return {
        "model": WaterQualityAnomalyDetector(contamination=0.05),
        "model_trained": False,
        "training_data": None,
        "current_data": None,
        "predictions": None,
        "anomaly_scores": None,
        "expected_values": None,
        "residuals": None,
    }


state = init_session()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_historical_data(site_id: str, start_year: int = 2025, end_year: int = 2026) -> pd.DataFrame:
    """Load historical water quality data for model training."""
    with st.spinner(f"Loading historical data from {start_year} to {end_year}..."):
        try:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"

            data, metadata = get_historical_data(
                site_id=site_id,
                start_date=start_date,
                end_date=end_date,
                days_back=None,
            )

            # Handle case where data might be a tuple or have specific structure
            if isinstance(data, tuple):
                data = data[0]

            df_formatted = format_data_for_modeling(data, metadata)

            if len(df_formatted) == 0:
                st.error("No historical data available for the selected date range.")
                return None

            # Remove any completely empty columns
            df_formatted = df_formatted.dropna(axis=1, how='all')

            st.success(
                f"✓ Loaded {len(df_formatted)} records from {df_formatted.index.min().date()} "
                f"to {df_formatted.index.max().date()}"
            )
            return df_formatted

        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return None


def load_current_data(site_id: str, hours: int = 24) -> pd.DataFrame:
    """Load current water quality data from the past N hours."""
    with st.spinner(f"Loading data from the past {hours} hours..."):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours)

            data, metadata = get_instantaneous_data(
                site_id=site_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

            # Handle case where data might be a tuple or have specific structure
            if isinstance(data, tuple):
                data = data[0]

            df_formatted = format_data_for_modeling(data, metadata)

            if len(df_formatted) == 0:
                st.warning(f"No data available for the past {hours} hours.")
                return None

            # Remove any completely empty columns
            df_formatted = df_formatted.dropna(axis=1, how='all')

            st.success(f"✓ Loaded {len(df_formatted)} current records")
            return df_formatted

        except Exception as e:
            st.error(f"Error loading current data: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return None


def train_model(df_train: pd.DataFrame):
    """Train the anomaly detection model."""
    with st.spinner("Training anomaly detection model..."):
        try:
            # Remove rows with any NaN values for training
            df_clean = df_train.dropna()

            if len(df_clean) < 100:
                st.error("Insufficient training data (minimum 100 records required).")
                return False

            state["model"].fit(df_clean)
            state["training_data"] = df_clean
            state["model_trained"] = True

            st.success(
                f"✓ Model trained on {len(df_clean)} records "
                f"({df_clean.index.min()} to {df_clean.index.max()})"
            )
            return True

        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False


def predict_anomalies(df_current: pd.DataFrame):
    """Predict anomalies in current data."""
    with st.spinner("Detecting anomalies..."):
        try:
            predictions, scores, expected, residuals = state["model"].predict(df_current)

            state["current_data"] = df_current
            state["predictions"] = predictions
            state["anomaly_scores"] = scores
            state["expected_values"] = expected
            state["residuals"] = residuals

            return True

        except Exception as e:
            st.error(f"Error predicting anomalies: {str(e)}")
            return False


def create_timeseries_plot(param_name: str) -> go.Figure:
    """Create an interactive time series plot with anomalies highlighted."""
    fig = go.Figure()

    actual = state["current_data"][param_name]
    expected = state["expected_values"][param_name]
    predictions = state["predictions"]

    # Add expected values as background
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=expected,
            mode="lines",
            name="Expected (Seasonal)",
            line=dict(color="lightblue", width=2),
            hovertemplate="<b>Expected</b><br>%{y:.2f}<br>%{x}<extra></extra>",
        )
    )

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            hovertemplate="<b>Actual</b><br>%{y:.2f}<br>%{x}<extra></extra>",
        )
    )

    # Highlight anomalies
    anomalies_mask = predictions == -1
    if anomalies_mask.any():
        fig.add_trace(
            go.Scatter(
                x=actual.index[anomalies_mask],
                y=actual.values[anomalies_mask],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
                hovertemplate="<b>⚠️ ANOMALY</b><br>%{y:.2f}<br>%{x}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{param_name} - Last 24 Hours",
        xaxis_title="Time",
        yaxis_title=param_name,
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def create_anomaly_score_plot() -> go.Figure:
    """Create a plot of anomaly scores over time."""
    fig = go.Figure()

    scores = state["anomaly_scores"]
    times = state["current_data"].index
    predictions = state["predictions"]

    # Add anomaly scores
    fig.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="purple", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            hovertemplate="<b>Score</b><br>%{y:.3f}<br>%{x}<extra></extra>",
        )
    )

    # Add threshold line
    threshold = -state["model"].model.offset_
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Anomaly Threshold",
        annotation_position="right",
    )

    fig.update_layout(
        title="Anomaly Scores - Last 24 Hours",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def create_parameter_comparison_plot() -> go.Figure:
    """Create a subplot comparing all parameters."""
    params = list(state["current_data"].columns)
    predictions = state["predictions"]
    anomalies_count = (predictions == -1).sum()

    # Create data for comparison
    comparison_data = []
    for param in params:
        actual = state["current_data"][param].copy()
        expected = state["expected_values"][param].copy()

        # Normalize to 0-100 scale for comparison
        actual_norm = (actual - actual.min()) / (actual.max() - actual.min()) * 100
        expected_norm = (expected - expected.min()) / (expected.max() - expected.min()) * 100

        comparison_data.append({
            "Parameter": param,
            "Min": actual.min(),
            "Max": actual.max(),
            "Mean": actual.mean(),
            "Std": actual.std(),
        })

    df_comparison = pd.DataFrame(comparison_data)

    fig = px.bar(
        df_comparison,
        x="Parameter",
        y=["Min", "Max", "Mean"],
        barmode="group",
        title="Parameter Statistics - Last 24 Hours",
        labels={"value": "Value", "variable": "Statistic"},
        height=400,
    )

    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Select monitoring station
    selected_station = st.selectbox(
        "Select Monitoring Station",
        options=list(MONITORING_STATIONS.keys()),
        format_func=lambda x: f"{MONITORING_STATIONS[x]['name']} ({x})",
    )

    st.divider()

    # Training data settings
    st.subheader("🎯 Model Training")

    col1, col2 = st.columns(2)
    with col1:
        training_start_year = st.number_input(
            "Training Start Year", min_value=2015, max_value=2026, value=2025
        )
    with col2:
        training_end_year = st.number_input(
            "Training End Year", min_value=2015, max_value=2026, value=2026
        )

    if training_start_year > training_end_year:
        st.error("Start year must be before end year")
        st.stop()

    # Contamination parameter
    contamination = st.slider(
        "Anomaly Contamination Rate",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Expected proportion of anomalies in the training data",
    )

    st.divider()

    # Current data period
    st.subheader("📊 Current Data")
    hours_back = st.slider(
        "Hours of Recent Data to Display",
        min_value=1,
        max_value=8760,
        value=24,
        step=1,
    )

    st.divider()

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        train_clicked = st.button("🔄 Train Model", use_container_width=True)

    with col2:
        analyze_clicked = st.button("📈 Analyze Data", use_container_width=True)


# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.title("💧 Water Quality Anomaly Detection Dashboard")

st.markdown(
    f"""
    **Station:** {MONITORING_STATIONS[selected_station]['name']}  
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
)

st.divider()

# TAB 1: TRAINING
if train_clicked:
    st.session_state.model_trained = False
    st.session_state.predictions = None

    # Load historical data
    df_train = load_historical_data(
        selected_station,
        start_year=training_start_year,
        end_year=training_end_year,
    )

    if df_train is not None:
        # Re-initialize model with new contamination parameter
        state["model"] = WaterQualityAnomalyDetector(
            contamination=contamination, random_state=42
        )

        # Train model
        if train_model(df_train):
            # Display training statistics
            st.subheader("📚 Training Data Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Records", f"{len(df_train):,}")

            with col2:
                duration = (df_train.index[-1] - df_train.index[0]).days
                st.metric("Days", duration)

            with col3:
                st.metric("Parameters", len(df_train.columns))

            with col4:
                st.metric("Missing", f"{df_train.isna().sum().sum():,}")

            # Display data summary
            st.subheader("Data Summary")
            st.dataframe(df_train.describe().T, use_container_width=True)

            st.success("✓ Model is ready for anomaly detection!")

# TAB 2: ANALYSIS
if analyze_clicked and not state["model_trained"]:
    st.warning("⚠️ Please train the model first using the 'Train Model' button.")

elif analyze_clicked and state["model_trained"]:
    # Load current data
    df_current = load_current_data(selected_station, hours=hours_back)

    if df_current is not None:
        # Predict anomalies
        if predict_anomalies(df_current):
            # Display key metrics
            st.subheader("⚠️ Anomaly Detection Results")

            predictions = state["predictions"]
            scores = state["anomaly_scores"]
            anomalies_count = (predictions == -1).sum()
            normal_count = (predictions == 1).sum()
            anomaly_pct = (anomalies_count / len(predictions)) * 100

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Normal Records", normal_count)

            with col2:
                st.metric("Anomalies Detected", anomalies_count, delta=f"{anomaly_pct:.1f}%")

            with col3:
                st.metric("Average Score", f"{scores.mean():.3f}")

            with col4:
                st.metric("Min/Max Score", f"{scores.min():.3f} / {scores.max():.3f}")

            st.divider()

            # CHARTS
            st.subheader("📊 Time Series Analysis")

            # Create only required tabs
            tab_ts, tab_details = st.tabs([
                "Time Series", "Anomaly Details"
            ])

            with tab_ts:
                # Plot each parameter
                params = list(state["current_data"].columns)

                for param in params:
                    fig = create_timeseries_plot(param)
                    st.plotly_chart(fig, use_container_width=True)

            with tab_details:
                st.subheader("Detailed Anomaly Information")

                if anomalies_count > 0:
                    # Find anomaly timestamps (most recent first)
                    anomaly_indices = np.where(predictions == -1)[0][::-1]

                    for idx in anomaly_indices[:10]:  # Show most recent 10 anomalies
                        timestamp = state["current_data"].index[idx]
                        residual_row = state["residuals"].iloc[idx]
                        actual_row = state["current_data"].iloc[idx]
                        expected_row = state["expected_values"].iloc[idx]

                        explanation = state["model"].explain_anomaly(
                            residual_row, actual_row, expected_row, top_n=3
                        )

                        with st.expander(
                            f"🚨 Anomaly at {timestamp.strftime('%Y-%m-%d %H:%M')} "
                            f"(Score: {scores[idx]:.3f})"
                        ):
                            st.dataframe(explanation, use_container_width=True)

                else:
                    st.info("✓ No anomalies detected in the selected time period.")

# Display model status in sidebar
st.sidebar.divider()
st.sidebar.subheader("📈 Model Status")

if state["model_trained"]:
    st.sidebar.success("✓ Model Trained")
    st.sidebar.caption(
        f"Training data: {len(state['training_data']):,} records "
        f"from {state['training_data'].index.min().strftime('%Y-%m-%d')} "
        f"to {state['training_data'].index.max().strftime('%Y-%m-%d')}"
    )
else:
    st.sidebar.warning("⚠️ Model Not Trained")
    st.sidebar.caption("Click 'Train Model' to get started")

if state["predictions"] is not None:
    anomalies = (state["predictions"] == -1).sum()
    st.sidebar.metric("Last Analysis", f"{anomalies} anomalies")
