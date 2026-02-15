# Water-Quality-Hazard-Prediction-and-Detection

# Project Overview
A work-in-progress machine learning system that monitors river water quality in real-time and detects anomalies that could indicate environmental hazards. It combines multiple anomaly detection techniques to identify potential pollution events, algal blooms, or other water quality issues before they become critical.

# Problem overview
Every year, industrial pollution, agricultural runoff, and natural events contaminate rivers across the United States, threatening public health, damaging ecosystems, and costing billions in economic losses.

**Reactive, Not Proactive**
   - Current monitoring relies on periodic sampling (often weekly or monthly)
   - By the time contamination is detected, it has already spread downstream
   - Communities are alerted **after** exposure, not before

**Fragmented Data**
   - Data sits in silos across federal, state, and local agencies
   - No real-time integration or automated analysis
   - Pattern recognition across different data sources is virtually impossible

**Resource Intensive**
   - Manual sampling costs $200-500 per sample
   - Lab analysis takes 3-5 days
   - Small municipalities can't afford comprehensive monitoring


# Data Sources - Description - Access - Frequency
https://waterservices.usgs.gov/ - Real Time - Free API - 15 to 60 Minutes

https://www.waterqualitydata.us/ - Historical Water Quality - Free Download - N/A

# Road Map
-Basic data collection from USGS

-Isolation Forest implementation

-Streamlit dashboard prototype
