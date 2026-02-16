import streamlit as st
import numpy as np
import plotly.graph_objects as go

from data import get_stock_data
from monte_carlo import simulate_price_paths, get_price_statistics

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Monte Carlo Simulator",
    layout="wide"
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0e1117;
    color: #f3f4f6;
}
.main-title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    background: linear-gradient(90deg, #3b82f6, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom:5px;
}
.subtitle {
    text-align:center;
    font-size:16px;
    color:#9ca3af;
    margin-bottom:25px;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding:18px;
    border-radius:16px;
    text-align:center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-label {
    font-size:13px;
    color:#9ca3af;
}
.metric-value {
    font-size:28px;
    font-weight:600;
}
.metric-delta {
    font-size:15px;
}
.section-title {
    font-size:20px;
    font-weight:600;
    margin-top:25px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-title">Monte Carlo Stock Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Probabilistic Forecasting • Geometric Brownian Motion</div>', unsafe_allow_html=True)

st.divider()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Simulation Controls")

ticker = st.sidebar.text_input("Stock Ticker", "NVDA", key="ticker_input").upper()
lookback_days = st.sidebar.slider("Lookback Window (Days)", 30, 252, 60, key="lookback_slider")
future_days = st.sidebar.slider("Forecast Horizon (Days)", 5, 180, 30, key="horizon_slider")
sims = st.sidebar.slider("Monte Carlo Simulations", 1000, 20000, 5000, step=1000, key="sims_slider")

vol_shock = st.sidebar.slider(
    label="Volatility Multiplier ",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Adjusts the volatility of the stock in the simulation.\n1.0 = baseline, <1 = less volatile, >1 = more volatile.",
    key="vol_slider"
)

# Dynamic caption for volatility
if vol_shock < 1.0:
    st.sidebar.write(f"Selected multiplier: {vol_shock:.1f}  Lower volatility, smoother price movements.")
elif vol_shock == 1.0:
    st.sidebar.write(f"Selected multiplier: {vol_shock:.1f}  Baseline volatility, normal market behavior.")
else:
    st.sidebar.write(f"Selected multiplier: {vol_shock:.1f}  Higher volatility, bigger swings & more risk/reward.")

st.sidebar.caption("Model assumes log-normal dynamics (GBM)")

# ==========================================
# LOAD DATA
# ==========================================
try:
    current_price, returns = get_stock_data(ticker, lookback_days)
    current_price = float(current_price)
    mu = float(returns.mean())
    sigma = float(returns.std())
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# ==========================================
# MONTE CARLO SIMULATION
# ==========================================
paths = simulate_price_paths(
    current_price,
    mu,
    sigma * vol_shock,
    future_days,
    sims
)

stats = get_price_statistics(paths)
mean_price = stats["mean"]
median_price = stats["median"]
p5 = stats["p5"]
p95 = stats["p95"]

# ==========================================
# ARROW LOGIC
# ==========================================
def arrow(expected, current):
    delta = expected - current
    if delta > 0:
        return "▲", "#22c55e", delta
    elif delta < 0:
        return "▼", "#ef4444", delta
    return "→", "#9ca3af", 0

mean_arrow, mean_color, mean_delta = arrow(mean_price, current_price)
median_arrow, median_color, median_delta = arrow(median_price, current_price)

# ==========================================
# METRIC CARDS
# ==========================================
st.markdown('<div class="section-title">Price Estimates</div>', unsafe_allow_html=True)
cols = st.columns(5)

def card(col, label, value, arrow_symbol=None, delta=None, color=None):
    if arrow_symbol:
        col.markdown(f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{color}">{arrow_symbol} {abs(delta):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col.markdown(f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

card(cols[0], "Current Price", f"${current_price:.2f}")
card(cols[1], "Expected Mean", f"${mean_price:.2f}", mean_arrow, mean_delta, mean_color)
card(cols[2], "Expected Median", f"${median_price:.2f}", median_arrow, median_delta, median_color)
card(cols[3], "5th Percentile", f"${p5:.2f}")
card(cols[4], "95th Percentile", f"${p95:.2f}")

# ==========================================
# RISK METRICS
# ==========================================
final_prices = paths[-1]
sim_returns = (final_prices - current_price) / current_price
expected_return = sim_returns.mean() * 100
sharpe_ratio = sim_returns.mean() / sim_returns.std() if sim_returns.std() != 0 else 0

st.markdown('<div class="section-title">Risk Metrics</div>', unsafe_allow_html=True)
rcols = st.columns(2)
card(rcols[0], "Expected Return", f"{expected_return:.2f}%")
card(rcols[1], "Sharpe Ratio", f"{sharpe_ratio:.2f}")

# ==========================================
# MONTE CARLO PATHS
# ==========================================
st.markdown('<div class="section-title">Monte Carlo Price Paths</div>', unsafe_allow_html=True)

median_path = np.median(paths, axis=1)
p5_path = np.percentile(paths, 5, axis=1)
p95_path = np.percentile(paths, 95, axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(y=median_path, mode="lines", name="Median", line=dict(color="#3b82f6", width=3)))
fig.add_trace(go.Scatter(y=p95_path, mode="lines", line=dict(color="rgba(59,130,246,0.3)"), showlegend=False))
fig.add_trace(go.Scatter(y=p5_path, mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,0.1)",
                         line=dict(color="rgba(59,130,246,0.3)"), showlegend=False))
fig.add_hline(y=current_price, line_dash="dash", line_color="#9ca3af")

fig.update_layout(height=450, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# DISTRIBUTION
# ==========================================
st.markdown('<div class="section-title">Final Price Distribution</div>', unsafe_allow_html=True)

hist = go.Figure()
hist.add_trace(go.Histogram(x=final_prices, nbinsx=60, marker_color="#3b82f6"))
hist.update_layout(height=400, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(hist, use_container_width=True)

st.divider()
st.markdown("""
---
**Disclaimer:**  
This application is provided for research and educational purposes only. 
The models and simulations are based on historical data and theoretical assumptions, 
which may not reflect future market conditions. Nothing displayed should be interpreted 
as financial advice or a recommendation to buy or sell any security. 
All investment decisions involve risk, including the potential loss of capital.
""")
