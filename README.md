# Water-Quality-Hazard-Prediction-and-Detection

# Project Overview
A work-in-progress machine learning system that monitors river water quality in real-time and detects anomalies that could indicate environmental hazards. It combines multiple anomaly detection techniques to identify potential pollution events, algal blooms, or other water quality issues before they become critical.

# Problem overview
Every year, industrial pollution, agricultural runoff, and natural events contaminate rivers across the United States, threatening public health, damaging ecosystems, and costing billions in economic losses.

**Reactive monitoring**
Most water quality monitoring relies on periodic sampling (e.g., weekly or monthly state programs), meaning rapid contamination events often go undetected until after people have been exposed.

**Fragmented Data**
Federal, state, and local agencies maintain separate monitoring networks with varying reporting frequencies, making real-time integration limited and pattern recognition difficult without custom data pipelines.

**Contamination Inflicts Substantial Economic Burden**
- Waterborne diseases alone account for billions in healthcare costs annually (Collier et al., 2021).
- Recreational water–associated illnesses carry $2.2 – $3.7 billion in economic burden (DeFlorio-Barker et al., 2018).
- Pollution can drive up drinking water treatment costs significantly (U.S. EPA, 2023).
- Broader ecological and economic impacts from nutrient pollution and eutrophication are estimated at billions annually (Dodds et al., 2009).

# Stakeholder Context 
Water contamination and delayed detection affect a broad set of stakeholders across the water and environmental ecosystem. Each group faces distinct operational risks, but all are constrained by limited real-time visibility and uncertainty.

## Stakeholder

Municipal Water Utility Directors and Operations Managers

**Pain Points**
- Drinking water utilities must ensure regulatory compliance under uncertainty about incoming source water quality

- To manage risk, utilities often apply conservative treatment strategies, including higher-than-necessary chemical dosing

- Sudden contamination events (e.g., nutrient spikes, turbidity increases, algal toxins) can force emergency treatment changes with little warning

- Treatment optimisation is limited by lagged data (manual sampling + lab turnaround)

**Operational Consequences**
- Increased chemical usage and energy costs

- Reduced ability to optimise treatment in real time

- Higher operational stress during rapid water quality changes

- Risk of either under-treatment (compliance violations) or over-treatment (unnecessary cost)

**What They Need**

- Real-time upstream water quality data

- Early warning signals before degraded water reaches intake points

- Predictive confidence to adjust treatment dynamically, not reactively

- Automated data records to support regulatory reporting and audits

**Supporting Sources**

U.S. Environmental Protection Agency — drinking water treatment and source water protection guidance

American Water Works Association — utility operations and treatment optimisation literature







## Stakeholder

State Environmental Agencies, Conservation Authorities

**Pain Points**

- Nutrient pollution and sediment runoff are episodic and spatially variable

- Manual sampling often misses short-duration events (e.g., post-storm runoff)

- Limited ability to assess cumulative or downstream impacts in real time

**Operational Consequences**

- Reactive enforcement and remediation

- Difficulty evaluating the effectiveness of nutrient reduction programs (periodic sampling)

- Delayed response to harmful algal bloom precursors

---

## 🚀 Using the Water Quality Anomaly Detection Dashboard

The Streamlit dashboard provides a user-friendly interface for training anomaly detection models and monitoring water quality in real time.

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will start and be available at `http://localhost:8501`

### Dashboard Features

#### 1. **Configuration Panel (Sidebar)**
- **Select Monitoring Station**: Choose from configured USGS monitoring stations
- **Model Training Settings**:
  - Set training data date range (default: 2022-2024)
  - Adjust contamination rate (expected proportion of anomalies, default: 5%)
- **Current Data Period**: Select how many hours of recent data to display (1-72 hours)

#### 2. **Model Training**
Click the **"🔄 Train Model"** button to:
- Fetch historical water quality data for the specified date range
- Train a seasonal anomaly detector that learns normal patterns
- Display training statistics and data summary

**What the Model Does:**
- Learns seasonal patterns for each water quality parameter (e.g., temperature, pH, dissolved oxygen)
- Uses an Isolation Forest algorithm to detect anomalies based on residuals from expected seasonal values
- Generates a baseline for what "normal" looks like

#### 3. **Anomaly Analysis**
Click the **"📈 Analyze Data"** button to:
- Load real-time water quality data from the past 24 hours
- Run anomaly detection using the trained model
- Display interactive visualizations

#### 4. **Visualizations**

**Time Series Tab**:
- Interactive line charts for each water quality parameter
- Shows actual values (blue line), expected seasonal values (light blue background), and anomalies (red X markers)
- Hover over points to see exact values and timestamps

**Anomaly Scores Tab**:
- Shows the anomaly detection confidence score for each time point
- Red dashed line indicates the anomaly threshold
- Higher positive scores = more likely to be normal; lower scores = anomalous

**Parameter Comparison Tab**:
- Bar chart comparing statistics (min, max, mean) across all parameters
- Useful for identifying which parameters are within expected ranges

**Anomaly Details Tab**:
- Expandable items for each detected anomaly
- Shows which parameters deviated most and their z-scores (how many standard deviations from expected)
- Displays actual vs. expected values for context

### Interpreting Results

**Anomaly Score Interpretation:**
- **Score > -0.5** (above threshold): Normal operation
- **Score between -0.5 and -2.0**: Mild anomaly
- **Score < -2.0**: Strong anomaly (high confidence)

**Parameter Details:**
When you expand an anomaly, you'll see:
- **value**: Actual measured value
- **expected**: What the model expected based on seasonal patterns
- **deviation**: Difference between actual and expected (residual)
- **z_score**: Standardized deviation (how many standard deviations from normal)

### Example Workflow

1. **Start Fresh**: Open the dashboard
2. **Configure**: Select your monitoring station and set training years (2022-2024)
3. **Train**: Click "Train Model" and wait for completion
4. **Analyze**: Click "Analyze Data" to see today's water quality with anomalies highlighted
5. **Investigate**: Click on anomalies to understand what deviated and by how much
6. **Adjust**: If too many false positives, increase hours_back or decrease contamination rate

### Model Performance Tips

- **Better Training Data**: More years of historical data = more robust seasonal patterns
- **Contamination Rate**: Lower values (e.g., 0.02) mean stricter anomaly detection; higher values (e.g., 0.10) are more lenient
- **Parameters**: The model learns which parameters are most correlated and uses all of them together for context

### Requirements

- Python 3.8+
- Streamlit 1.0+
- pandas, sklearn, plotly
- USGS dataretrieval library (for water quality data)
- Internet connection (to fetch USGS data)

### Troubleshooting

**"No data available" error:**
- Check internet connectivity
- Verify the monitoring station ID is valid
- The USGS API may have rate limits; try again later

**"Model must be fitted first" error:**
- Train the model before analyzing data (click "Train Model" first)

**False Positives / Sudden Pattern Jumps:**
- **Symptom**: Expected values suddenly jump, causing all recent data to appear as anomalies
- **Causes**:
  - Seasonal patterns have gaps or sparse data for recent day-of-year
  - Training data doesn't cover all days of year uniformly
  - Interpolation creates artifacts at pattern boundaries
  
- **Solutions**:
  1. **Lower the contamination rate** to 0.01-0.02 (stricter detection reduces false positives)
  2. **Use more recent training data** (e.g., 2025-2026) closer to current date
  3. **Increase training span** (use 3-4 years of data for smoother patterns)
  4. **Train on consistent periods** (avoid years with extreme events that skew patterns)

**Slow startup:**
- Large date ranges can take time to fetch and process
- Try reducing the training date range initially

**What They Need**

- Continuous, high-frequency water quality signals

- Early indicators of eutrophication and ecological stress










# Data Sources - Description - Access - Frequency
https://waterservices.usgs.gov/ - Real Time - Free API - 15 to 60 Minutes

https://www.waterqualitydata.us/ - Historical Water Quality - Free Download - N/A

# Road Map
-Basic data collection from USGS

-Isolation Forest implementation

-Streamlit dashboard prototype

# References
Centers for Disease Control and Prevention (CDC).
Collier, S. A., Deng, L., Adam, E. A., Benedict, K. M., Beshearse, E. M., Blackstock, A. J., … Yoder, J. S. (2021).
Estimate of burden and direct healthcare cost of infectious waterborne disease in the United States.
Emerging Infectious Diseases, 27(1), 140–149.
https://doi.org/10.3201/eid2701.190676

Centers for Disease Control and Prevention (CDC).
DeFlorio-Barker, S., Wing, C., Jones, R. M., Dorevitch, S., & Wade, T. J. (2018).
Estimated economic burden of recreational waterborne illness in the United States.
American Journal of Public Health, 108(2), 256–263.
https://doi.org/10.2105/AJPH.2017.304174

U.S. Environmental Protection Agency (EPA).
(2023).
The effects of nutrient pollution on the economy.
https://www.epa.gov/nutrientpollution/effects-economy

Dodds, W. K., Bouska, W. W., Eitzmann, J. L., Pilger, T. J., Pitts, K. L., Riley, A. J., … Thornbrugh, D. J. (2009).
Eutrophication of U.S. freshwaters: Analysis of potential economic damages.
Environmental Science & Technology, 43(1), 12–19.
https://doi.org/10.1021/es801217q

National Oceanic and Atmospheric Administration (NOAA).
(2019).
Hitting us where it hurts: The untold story of harmful algal blooms.
https://www.fisheries.noaa.gov/feature-story/hitting-us-where-it-hurts-untold-story-harmful-algal-blooms

U.S. Environmental Protection Agency.
(2023). Drinking water treatment and source water protection.
https://www.epa.gov/dwreginfo
https://www.epa.gov/sourcewaterprotection

American Water Works Association.
(2021). Water quality and treatment: A handbook on drinking water (7th ed.).
McGraw-Hill Education.
