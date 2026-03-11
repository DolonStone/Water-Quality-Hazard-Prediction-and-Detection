from src.data import get_instantaneous_data, get_historical_data,format_data_for_modeling
from src.models import WaterQualityAnomalyDetector

def test_anomaly_detector():
    # Initialize the anomaly detector
    detector = WaterQualityAnomalyDetector()
    
    # Train model on historical data (using a known site with good data)
    historical_df, historical_metadata = get_historical_data(site_id='USGS-07374000', start_date='2022-01-01', end_date='2023-01-01')
    
    historical_df_formatted = format_data_for_modeling(historical_df, historical_metadata)
    detector.fit(historical_df_formatted)
    # Get some test data (using a known site with good data)
    df, metadata = get_instantaneous_data(site_id='USGS-07374000')
    df = format_data_for_modeling(df, metadata)
    # Run anomaly detection
    predictions,anomaly_scores = detector.predict(df)
    print(predictions.size)
    
    df['prediction'] = predictions
    df['anomaly_score'] = anomaly_scores
    df[df['prediction'] == -1]  # rows flagged as anomalies


test_anomaly_detector()