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

- Difficulty evaluating the effectiveness of nutrient reduction programs

- Delayed response to harmful algal bloom precursors

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
