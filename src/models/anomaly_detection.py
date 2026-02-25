import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class WaterQualityAnomalyDetector:
    """
    Isolation Forest-based anomaly detector for water quality parameters.
    """
    
    def __init__(self, contamination=0.05, random_state=42,use_model = "IsolationForest"):
        """
        Args:
            contamination: Expected proportion of anomalies (5% default)
            random_state: For reproducibility
        
        """
        if use_model == "IsolationForest":
            self.model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100
            )
        else:
            raise ValueError("Please provide valid model\nCurrent Valid Models:\n Isolation Forest")
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, df):
        """
        Train the anomaly detector on historical data.
        
        Args:
            df: DataFrame with water quality features
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(df)
        
        # Train model
        self.model.fit(X_scaled)
        self.feature_names = df.columns.tolist()
        self.is_fitted = True
        
        print(f"Model trained on {len(df)} samples with {len(self.feature_names)} features")
        return self
    
    def predict(self, df):
        """
        Predict anomalies on new data.
        
        Args:
            df: DataFrame with same features as training data
            
        Returns:
            predictions: -1 for anomaly, 1 for normal
            anomaly_scores: Anomaly score (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(df)
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        return predictions, anomaly_scores
    
    def save(self, filepath):
        """Save trained model and scaler."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
        return self