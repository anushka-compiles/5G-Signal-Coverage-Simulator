
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="5G Signal Coverage Simulator",
    page_icon="📡",
    layout="wide"
)

# ==========================================
# 2. PROJECT TITLE
# ==========================================

st.title("📡 5G Signal Strength and Coverage Prediction Simulator")

st.markdown("""
This application simulates a virtual 5G base station and predicts
received signal strength at different distances using the
Free Space Path Loss (FSPL) model.

**Note:** This is a theoretical software simulation, not a
measurement of an actual 5G network.
""")

st.divider()

# ==========================================
# 3. SIDEBAR - NETWORK PARAMETERS
# ==========================================

st.sidebar.header("📶 Network Parameters")

frequency = st.sidebar.selectbox(
    "5G Frequency (MHz)",
    [700, 1800, 3500, 28000],
    index=2
)

transmit_power = st.sidebar.slider(
    "Transmitting Power (dBm)",
    min_value=0,
    max_value=50,
    value=30
)

tx_gain = st.sidebar.slider(
    "Transmitter Antenna Gain (dBi)",
    min_value=0,
    max_value=25,
    value=15
)

rx_gain = st.sidebar.slider(
    "Receiver Antenna Gain (dBi)",
    min_value=0,
    max_value=20,
    value=5
)

max_distance = st.sidebar.slider(
    "Simulation Distance (km)",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.divider()

st.sidebar.info(
    "Change the network parameters to observe "
    "their effect on predicted 5G coverage."
)

# ==========================================
# 4. MATHEMATICAL MODEL
# ==========================================

def calculate_fspl(frequency_mhz, distance_km):
    """
    Calculate Free Space Path Loss.

    Frequency is in MHz.
    Distance is in kilometres.
    """

    safe_distance = np.maximum(distance_km, 0.001)

    fspl = (
        32.44
        + 20 * np.log10(frequency_mhz)
        + 20 * np.log10(safe_distance)
    )

    return fspl


def calculate_received_power(
    transmit_power,
    tx_gain,
    rx_gain,
    fspl
):
    """
    Calculate received signal power in dBm.
    """

    received_power = (
        transmit_power
        + tx_gain
        + rx_gain
        - fspl
    )

    return received_power


# ==========================================
# 5. SIGNAL CLASSIFICATION
# ==========================================

def classify_signal(power):
    """
    Classify predicted signal strength.
    """

    if power >= -70:
        return "Excellent"

    elif power >= -85:
        return "Good"

    elif power >= -100:
        return "Fair"

    else:
        return "Poor"


# ==========================================
# 6. DISTANCE-BASED SIMULATION
# ==========================================

distances = np.linspace(
    0.05,
    max_distance,
    100
)

fspl_values = calculate_fspl(
    frequency,
    distances
)

received_values = calculate_received_power(
    transmit_power,
    tx_gain,
    rx_gain,
    fspl_values
)

classifications = [
    classify_signal(power)
    for power in received_values
]

# Create prediction table

results = pd.DataFrame({
    "Distance (km)": distances,
    "FSPL (dB)": fspl_values,
    "Received Power (dBm)": received_values,
    "Signal Quality": classifications
})

# ==========================================
# 7. PERFORMANCE METRICS
# ==========================================

reference_distance = 1.0

reference_fspl = calculate_fspl(
    frequency,
    reference_distance
)

reference_power = calculate_received_power(
    transmit_power,
    tx_gain,
    rx_gain,
    reference_fspl
)

maximum_power = received_values[-1]

maximum_fspl = fspl_values[-1]

overall_quality = classify_signal(reference_power)

# ==========================================
# 8. DASHBOARD METRICS
# ==========================================

st.header("📊 Network Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Operating Frequency",
    f"{frequency} MHz"
)

col2.metric(
    "FSPL at 1 km",
    f"{reference_fspl:.2f} dB"
)

col3.metric(
    "Received Power at 1 km",
    f"{reference_power:.2f} dBm"
)

col4.metric(
    "Signal Quality at 1 km",
    overall_quality
)

st.caption(
    f"Predicted received power at the maximum "
    f"simulation distance of {max_distance} km: "
    f"{maximum_power:.2f} dBm"
)

st.divider()

# ==========================================
# 9. SIGNAL STRENGTH VS DISTANCE GRAPH
# ==========================================

st.header("📈 Signal Strength vs Distance")

signal_df = pd.DataFrame({
    "Distance (km)": distances,
    "Received Power (dBm)": received_values
})

fig_signal = px.line(
    signal_df,
    x="Distance (km)",
    y="Received Power (dBm)",
    title="Predicted 5G Signal Strength vs Distance",
    markers=True
)

fig_signal.update_layout(
    xaxis_title="Distance from Base Station (km)",
    yaxis_title="Received Signal Power (dBm)",
    template="plotly_white"
)

st.plotly_chart(
    fig_signal,
    use_container_width=True
)

st.caption(
    "The predicted signal strength decreases as the "
    "distance from the base station increases."
)

st.divider()

# ==========================================
# 10. TWO-DIMENSIONAL COVERAGE HEATMAP
# ==========================================

st.header("🌐 2D 5G Coverage Heatmap")

grid_points = 100

x = np.linspace(
    -max_distance,
    max_distance,
    grid_points
)

y = np.linspace(
    -max_distance,
    max_distance,
    grid_points
)

X, Y = np.meshgrid(x, y)

# Distance from the virtual base station

R = np.sqrt(X**2 + Y**2)

# Calculate signal strength at every grid point

heatmap_fspl = calculate_fspl(
    frequency,
    R
)

heatmap_power = calculate_received_power(
    transmit_power,
    tx_gain,
    rx_gain,
    heatmap_fspl
)

fig_heatmap = px.imshow(
    heatmap_power,
    x=x,
    y=y,
    origin="lower",
    color_continuous_scale="RdYlGn",
    zmin=-100,
    zmax=-50,
    aspect="equal",
    labels={
        "x": "X Distance (km)",
        "y": "Y Distance (km)",
        "color": "Signal (dBm)"
    },
    title="Predicted 5G Signal Coverage Around Base Station"
)

fig_heatmap.update_layout(
    template="plotly_white"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

st.caption(
    "The virtual base station is located at the centre (0, 0). "
    "The heatmap represents an idealized, circular free-space "
    "coverage pattern."
)

st.divider()

# ==========================================
# 11. SIGNAL QUALITY CLASSIFICATION CHART
# ==========================================

st.header("📊 Signal Quality Classification")

quality_order = [
    "Excellent",
    "Good",
    "Fair",
    "Poor"
]

quality_counts = (
    results["Signal Quality"]
    .value_counts()
    .reindex(quality_order, fill_value=0)
    .reset_index()
)

quality_counts.columns = [
    "Signal Quality",
    "Number of Samples"
]

fig_quality = px.bar(
    quality_counts,
    x="Signal Quality",
    y="Number of Samples",
    color="Signal Quality",
    title="Signal Quality Distribution Across Simulated Distances",
    category_orders={
        "Signal Quality": quality_order
    }
)

fig_quality.update_layout(
    template="plotly_white",
    showlegend=False
)

st.plotly_chart(
    fig_quality,
    use_container_width=True
)

st.divider()

# ==========================================
# 12. PREDICTION DATA TABLE
# ==========================================

st.header("📋 Signal Prediction Results")

st.dataframe(
    results.round(3),
    use_container_width=True
)

# ==========================================
# 13. CSV DOWNLOAD
# ==========================================

csv_data = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Prediction Results (CSV)",
    data=csv_data,
    file_name="5G_Signal_Prediction_Results.csv",
    mime="text/csv"
)

st.divider()

# ==========================================
# 14. PROJECT INFORMATION
# ==========================================

st.header("ℹ️ About the Project")

st.markdown("""
### 5G Signal Strength and Coverage Prediction Simulator

**Project Type:** Individual ECE Software Simulation Project

**Programming Language:** Python

**Libraries Used:**
- Streamlit
- NumPy
- Pandas
- Plotly

**Simulation Model:** Free Space Path Loss (FSPL)

### Main Objectives

1. Simulate a virtual 5G base station.
2. Predict signal strength at different distances.
3. Calculate free-space path loss.
4. Visualize predicted coverage using a 2D heatmap.
5. Classify signal quality.
6. Export simulation results as a CSV file.

### Limitations

This application uses an ideal free-space propagation model.
It does not account for buildings, walls, terrain, atmospheric
effects, interference, shadowing or multipath fading.

The results are theoretical predictions and are not actual
5G network measurements.
""")

st.divider()

st.caption(
    "5G Signal Strength and Coverage Prediction Simulator | "
    "ECE Individual Project"
)
