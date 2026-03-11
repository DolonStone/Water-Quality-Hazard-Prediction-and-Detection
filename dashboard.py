"""
Water Quality Hazard Detection Dashboard
Real-time monitoring with anomaly detection using Isolation Forest.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data import get_instantaneous_data, get_historical_data, format_data_for_modeling
from src.models import WaterQualityAnomalyDetector
from src.data.station_config import MONITORING_STATIONS, WATER_QUALITY_PARAMS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0a0f1e;
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #38bdf8 !important;
}

.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #0284c7);
}

.metric-card.anomaly {
    border-color: #ef4444;
    animation: pulse-border 1.5s infinite;
}

.metric-card.anomaly::before {
    background: linear-gradient(90deg, #ef4444, #dc2626);
}

@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 8px 0 4px;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.anomaly-badge {
    display: inline-block;
    background: #ef4444;
    color: white;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

.normal-badge {
    display: inline-block;
    background: #10b981;
    color: white;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

.status-bar {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 16px;
}

.alert-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
}

.alert-box .alert-time {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #ef4444;
}

.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0369a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.sidebar .sidebar-content {
    background: #0f172a;
}

.stSidebar {
    background-color: #0f172a !important;
}

hr {
    border-color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'live_df' not in st.session_state:
    st.session_state.live_df = None
if 'expected_df' not in st.session_state:
    st.session_state.expected_df = None
if 'residuals_df' not in st.session_state:
    st.session_state.residuals_df = None
if 'anomaly_log' not in st.session_state:
    st.session_state.anomaly_log = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'last_retrain' not in st.session_state:
    st.session_state.last_retrain = None
if 'selected_anomaly_ts' not in st.session_state:
    st.session_state.selected_anomaly_ts = None

RETRAIN_INTERVAL_HOURS = 6
REFRESH_INTERVAL_MINS = 15

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=RETRAIN_INTERVAL_HOURS * 3600, show_spinner=False)
def load_historical(site_id: str, train_start: str, train_end: str):
    """Fetch and format historical data. Cached for 6 hours."""
    df_raw, metadata = get_historical_data(
        site_id=site_id,
        start_date=train_start,
        end_date=train_end,
    )
    return format_data_for_modeling(df_raw, metadata)


def load_and_train(site_id: str, train_start: str, train_end: str):
    """Load historical data and train anomaly detector."""
    df_formatted = load_historical(site_id, train_start, train_end)
    detector = WaterQualityAnomalyDetector(contamination=0.05)
    detector.fit(df_formatted)
    return detector, df_formatted


def fetch_live(site_id: str):
    """Fetch last 24h of data."""
    df_raw, metadata = get_instantaneous_data(site_id=site_id)
    return format_data_for_modeling(df_raw, metadata)


def run_detection(detector, df):
    predictions, scores, expected_df, residuals_df = detector.predict(df)
    df = df.copy()
    df['anomaly'] = predictions
    df['anomaly_score'] = scores
    return df, expected_df, residuals_df


def make_param_chart(df, param, title, expected_df=None):
    """Build a time-series chart for one parameter with anomaly markers and expected band."""

    if param not in df.columns:
        return None

    anomalies = df[df['anomaly'] == -1]
    normals = df[df['anomaly'] == 1]

    fig = go.Figure()

    # ── Expected band ───────────────────────────────────────────
    if expected_df is not None and param in expected_df.columns:

        expected = expected_df[param]

        # residual noise level
        std = (df[param] - expected).std()

        upper = expected + std
        lower = expected - std

        fig.add_trace(go.Scatter(
            x=list(df.index) + list(df.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(56,189,248,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=expected,
            mode='lines',
            line=dict(color='rgba(56,189,248,0.45)', width=1.5, dash='dot'),
            hovertemplate='Expected: %{y:.2f}<extra></extra>',
            name='Expected'
        ))

    # ── Normal readings ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=normals.index,
        y=normals[param],
        mode='lines',
        name='Normal',
        line=dict(color='#38bdf8', width=1.5),
        hovertemplate='%{x}<br>%{y:.2f}<extra>Normal</extra>',
    ))

    # ── Anomalies ───────────────────────────────────────────────
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies[param],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='#ef4444',
                size=9,
                symbol='circle',
                line=dict(color='#fca5a5', width=1.5)
            ),
            hovertemplate='%{x}<br>%{y:.2f}<extra>⚠ ANOMALY</extra>',
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(family='Space Mono', size=13, color='#94a3b8')),
        plot_bgcolor='#0a0f1e',
        paper_bgcolor='#0a0f1e',
        font=dict(color='#64748b', family='DM Sans'),
        xaxis=dict(gridcolor='#1e293b', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#1e293b', showgrid=True, zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=220,
    )

    return fig


def make_score_chart(df):
    """Anomaly score timeline."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['anomaly_score'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.08)',
        line=dict(color='#38bdf8', width=1),
        name='Score',
        hovertemplate='%{x}<br>Score: %{y:.3f}<extra></extra>',
    ))

    # Threshold line (approximate)
    threshold = df['anomaly_score'].quantile(0.05)
    fig.add_hline(y=threshold, line_dash='dot', line_color='#ef4444',
                  annotation_text='Anomaly threshold', annotation_font_color='#ef4444')

    fig.update_layout(
        title=dict(text='Anomaly Score Timeline (lower = more anomalous)',
                   font=dict(family='Space Mono', size=13, color='#94a3b8')),
        plot_bgcolor='#0a0f1e',
        paper_bgcolor='#0a0f1e',
        font=dict(color='#64748b', family='DM Sans'),
        xaxis=dict(gridcolor='#1e293b', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#1e293b', showgrid=True, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=200,
        showlegend=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 WQ Monitor")
    st.markdown("---")

    site_options = {v['name']: k for k, v in MONITORING_STATIONS.items()}
    selected_name = st.selectbox("Monitoring Station", list(site_options.keys()))
    selected_site = site_options[selected_name]

    st.markdown("---")
    st.markdown("**Model Settings**")
    contamination = st.slider("Anomaly sensitivity", 0.01, 0.15, 0.05, 0.01,
                               help="Expected proportion of anomalies. Higher = more sensitive.")

    st.markdown("**Training Period**")
    training_mode = st.radio(
        "Define by",
        ["Days back", "Date range"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if training_mode == "Days back":
        days_back = st.slider("Days of history", min_value=30, max_value=1000, value=500, step=10,
                              help="How many days of historical data to train on.")
        train_start = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        train_end = datetime.now().strftime('%Y-%m-%d')
    else:
        col_s, col_e = st.columns(2)
        with col_s:
            train_start = st.date_input("From", value=datetime.now() - timedelta(days=500),
                                         max_value=datetime.now() - timedelta(days=7))
            train_start = train_start.strftime('%Y-%m-%d')
        with col_e:
            train_end = st.date_input("To", value=datetime.now(),
                                       max_value=datetime.now())
            train_end = train_end.strftime('%Y-%m-%d')
        days_back = (datetime.strptime(train_end, '%Y-%m-%d') - datetime.strptime(train_start, '%Y-%m-%d')).days

    st.markdown(f"<div style='font-family:Space Mono;font-size:0.7rem;color:#64748b;margin-top:4px'>{train_start} → {train_end} ({days_back}d)</div>", unsafe_allow_html=True)

    st.markdown("---")

    train_col, refresh_col = st.columns(2)
    with train_col:
        train_btn = st.button("🔁 Train", use_container_width=True)
    with refresh_col:
        refresh_btn = st.button("↻ Refresh", use_container_width=True)

    st.markdown("---")

    if st.session_state.last_retrain:
        st.markdown(f"<div class='metric-label'>Last retrain</div><div style='font-family:Space Mono;font-size:0.75rem;color:#38bdf8'>{st.session_state.last_retrain.strftime('%H:%M %d %b')}</div>", unsafe_allow_html=True)
    if st.session_state.last_refresh:
        st.markdown(f"<div class='metric-label' style='margin-top:8px'>Last refresh</div><div style='font-family:Space Mono;font-size:0.75rem;color:#38bdf8'>{st.session_state.last_refresh.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='metric-label'>Auto-refresh every</div><div style='font-family:Space Mono;font-size:0.75rem;color:#64748b'>{REFRESH_INTERVAL_MINS} min</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-label' style='margin-top:6px'>Model retrains every</div><div style='font-family:Space Mono;font-size:0.75rem;color:#64748b'>{RETRAIN_INTERVAL_HOURS} hrs</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Water Quality Hazard Monitor")
station_info = MONITORING_STATIONS[selected_site]
st.markdown(f"<div class='status-bar'>📍 {station_info['name']} &nbsp;|&nbsp; Site: {selected_site} &nbsp;|&nbsp; State: {station_info['state_code']}</div>", unsafe_allow_html=True)

# ── Train model ───────────────────────────────────────────────────────────────
should_retrain = (
    train_btn or
    st.session_state.detector is None or
    (st.session_state.last_retrain and
     datetime.now() - st.session_state.last_retrain > timedelta(hours=RETRAIN_INTERVAL_HOURS))
)

if should_retrain:
    with st.spinner(f"Training model on historical data ({train_start} → {train_end})..."):
        try:
            detector, _ = load_and_train(selected_site, train_start, train_end)
            st.session_state.detector = detector
            st.session_state.model_trained = True
            st.session_state.last_retrain = datetime.now()
            st.success("✅ Model trained successfully.")
        except Exception as e:
            st.error(f"Training failed: {e}")

# ── Fetch live data ───────────────────────────────────────────────────────────
should_refresh = (
    refresh_btn or
    st.session_state.live_df is None or
    (st.session_state.last_refresh and
     datetime.now() - st.session_state.last_refresh > timedelta(minutes=REFRESH_INTERVAL_MINS))
)

if should_refresh and st.session_state.detector is not None:
    with st.spinner("Fetching live data..."):
        try:
            live_df = fetch_live(selected_site)
            result_df, expected_df, residuals_df = run_detection(st.session_state.detector, live_df)
            st.session_state.live_df = result_df
            st.session_state.expected_df = expected_df
            st.session_state.residuals_df = residuals_df
            st.session_state.last_refresh = datetime.now()

            # Log new anomalies
            new_anomalies = result_df[result_df['anomaly'] == -1]
            for ts, row in new_anomalies.iterrows():
                entry = {'time': ts, 'score': row['anomaly_score'], 'site': selected_site}
                if entry not in st.session_state.anomaly_log:
                    st.session_state.anomaly_log.append(entry)
            # Keep last 50
            st.session_state.anomaly_log = st.session_state.anomaly_log[-50:]

        except Exception as e:
            st.error(f"Live data fetch failed: {e}")

# ── Main dashboard ────────────────────────────────────────────────────────────
if st.session_state.live_df is not None:
    df = st.session_state.live_df

    # Metric cards row
    param_cols = [c for c in df.columns if c in WATER_QUALITY_PARAMS.values()]
    latest = df.iloc[-1] if len(df) > 0 else None

    if latest is not None:
        cols = st.columns(len(param_cols))
        for i, param in enumerate(param_cols):
            is_anomaly = latest.get('anomaly', 1) == -1
            card_class = "metric-card anomaly" if is_anomaly else "metric-card"
            badge = '<span class="anomaly-badge">⚠ anomaly</span>' if is_anomaly else '<span class="normal-badge">● normal</span>'
            val = latest[param]
            val_str = f"{val:.2f}" if not pd.isna(val) else "N/A"
            with cols[i]:
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="metric-label">{param}</div>
                    <div class="metric-value">{val_str}</div>
                    {badge}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary stats row
    total = len(df)
    n_anomalies = (df['anomaly'] == -1).sum()
    pct = (n_anomalies / total * 100) if total > 0 else 0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total readings (24h)", total)
    with s2:
        st.metric("Anomalies detected", n_anomalies)
    with s3:
        st.metric("Anomaly rate", f"{pct:.1f}%")
    with s4:
        latest_score = df['anomaly_score'].iloc[-1] if len(df) > 0 else 0
        st.metric("Latest anomaly score", f"{latest_score:.3f}")

    st.markdown("---")

    # Parameter charts
    st.markdown("### Parameter Time Series")
    chart_cols = st.columns(2)
    for i, param in enumerate(param_cols):
        fig = make_param_chart(df, param, param, expected_df=st.session_state.expected_df)
        if fig:
            with chart_cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

    # Anomaly score timeline
    st.markdown("### Anomaly Score")
    st.plotly_chart(make_score_chart(df), use_container_width=True)

    # Anomaly log + inspector
    st.markdown("---")
    st.markdown("### Recent Anomaly Events")
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:12px'>Click an anomaly to inspect which parameter likely caused it.</div>", unsafe_allow_html=True)

    anomaly_rows = df[df['anomaly'] == -1]

    if len(anomaly_rows) == 0:
        st.markdown("<div style='color:#64748b;font-size:0.9rem'>No anomalies detected in the current window.</div>", unsafe_allow_html=True)
    else:
        # Show anomaly list as clickable buttons
        for ts, row in anomaly_rows.iloc[::-1].iterrows():
            score = row['anomaly_score']
            label = f"⚠  {ts.strftime('%Y-%m-%d %H:%M')}  —  score: {score:.4f}"
            is_selected = st.session_state.selected_anomaly_ts == str(ts)
            btn_style = "background:rgba(239,68,68,0.2);border:1px solid #ef4444;" if is_selected else "background:rgba(239,68,68,0.07);border:1px solid #7f1d1d;"
            col_btn, col_expand = st.columns([5, 1])
            with col_btn:
                st.markdown(f"""
                <div style="{btn_style}border-radius:8px;padding:10px 14px;margin:4px 0;font-family:Space Mono;font-size:0.78rem;color:#fca5a5;">
                    {label}
                </div>""", unsafe_allow_html=True)
            with col_expand:
                if st.button("Inspect", key=f"inspect_{ts}"):
                    if st.session_state.selected_anomaly_ts == str(ts):
                        st.session_state.selected_anomaly_ts = None  # toggle off
                    else:
                        st.session_state.selected_anomaly_ts = str(ts)

            # Show explanation inline if this anomaly is selected
            if st.session_state.selected_anomaly_ts == str(ts):
                residual_row = st.session_state.residuals_df.loc[ts] if ts in st.session_state.residuals_df.index else pd.Series()
                expected_row = st.session_state.expected_df.loc[ts] if ts in st.session_state.expected_df.index else pd.Series()
                explanation = st.session_state.detector.explain_anomaly(residual_row, row, expected_row)
                st.markdown(f"""
                <div style="background:#0f172a;border:1px solid #1e3a5f;border-left:4px solid #38bdf8;
                            border-radius:8px;padding:16px;margin:4px 0 12px 0;">
                    <div style="font-family:Space Mono;font-size:0.75rem;color:#38bdf8;margin-bottom:12px">
                        ROOT CAUSE ANALYSIS — {ts.strftime('%Y-%m-%d %H:%M')}
                    </div>
                """, unsafe_allow_html=True)

                for rank, exp_row in explanation.iterrows():
                    z = exp_row['z_score']
                    direction = "↑ above" if exp_row['deviation'] > 0 else "↓ below"
                    severity_color = "#ef4444" if abs(z) > 3 else "#f97316" if abs(z) > 2 else "#eab308"
                    bar_width = min(int(abs(z) / 5 * 100), 100)
                    st.markdown(f"""
                    <div style="margin-bottom:12px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span style="color:#f1f5f9;font-size:0.85rem;font-weight:500">{exp_row['parameter']}</span>
                            <span style="font-family:Space Mono;font-size:0.75rem;color:{severity_color}">z = {z:+.2f}</span>
                        </div>
                        <div style="background:#1e293b;border-radius:4px;height:6px;margin-bottom:4px">
                            <div style="background:{severity_color};width:{bar_width}%;height:6px;border-radius:4px;transition:width 0.3s"></div>
                        </div>
                        <div style="font-size:0.75rem;color:#64748b">
                            Measured <span style="color:#f1f5f9">{exp_row['value']}</span>
                            &nbsp;{direction} expected&nbsp;
                            <span style="color:#94a3b8">{exp_row['expected']}</span>
                            &nbsp;(Δ {exp_row['deviation']:+.3f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.detector is None:
    st.info("👈 Click **Train** in the sidebar to initialise the model and begin monitoring.")
else:
    st.info("👈 Click **Refresh** to fetch live data.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#1e293b;font-family:Space Mono;font-size:0.7rem'>Auto-refreshing every {REFRESH_INTERVAL_MINS} minutes</div>", unsafe_allow_html=True)
time.sleep(0)
st.rerun() if (
    st.session_state.last_refresh and
    datetime.now() - st.session_state.last_refresh > timedelta(minutes=REFRESH_INTERVAL_MINS)
) else None