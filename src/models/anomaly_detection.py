import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


class WaterQualityAnomalyDetector:
    """
    Water quality anomaly detector using seasonal residual modeling.

    Expected values are learned from historical seasonal patterns using:
        (15-minute slot of day, day of year)

    That gives 96 * 365 seasonal positions.

    Isolation Forest is then trained on residuals (actual - expected).
    """

    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150
        )

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

        self._patterns = {}

        self._residual_mean = None
        self._residual_std = None

    # -------------------------------------------------------
    # Seasonal pattern builder
    # -------------------------------------------------------

    def _build_pattern(self, series: pd.Series):

        df = pd.DataFrame({
            "value": series,
            "slot": series.index.hour * 4 + series.index.minute // 15,
            "doy": series.index.dayofyear
        })

        min_required = 14 * 96

        if len(series) >= min_required:

            pattern = df.groupby(["slot", "doy"])["value"].mean()
            pattern = pattern.unstack(level="doy")

            # Wrap edges before smoothing so boundary DOYs aren't under-averaged
            pad = 15
            cols = sorted(pattern.columns)
            left_cols = cols[-pad:]   # last N DOYs of year
            right_cols = cols[:pad]   # first N DOYs of year

            left_pad = pattern[left_cols].copy()
            left_pad.columns = [c - 365 for c in left_cols]   # shift to negative DOYs

            right_pad = pattern[right_cols].copy()
            right_pad.columns = [c + 365 for c in right_cols]  # shift past 365

            padded = pd.concat([left_pad, pattern, right_pad], axis=1).sort_index(axis=1)

            # Smooth across days on padded pattern
            smoothed = padded.T.rolling(
                window=7,
                center=True,
                min_periods=1
            ).mean().T

            # Drop the padding columns, keep only real DOYs
            pattern = smoothed[cols]

            # Smooth across time-of-day
            pattern = pattern.rolling(
                window=3,
                center=True,
                min_periods=1
            ).mean()

            # Fill missing values
            slot_mean = df.groupby("slot")["value"].mean()

            for s in range(96):

                if s not in pattern.index:
                    pattern.loc[s] = slot_mean.get(s, np.nan)

                pattern.loc[s] = pattern.loc[s].interpolate(
                    method="linear",
                    limit_direction="both"
                )

                if pattern.loc[s].isna().any():
                    pattern.loc[s] = pattern.loc[s].fillna(slot_mean.get(s))

            pattern = pattern.sort_index()

            return pattern

        # fallback only if dataset extremely small
        slot_pattern = df.groupby("slot")["value"].mean()

        slot_pattern = slot_pattern.rolling(
            window=5,
            center=True,
            min_periods=1
        ).mean()

        return slot_pattern

    # -------------------------------------------------------
    # Expected value lookup
    # -------------------------------------------------------


    def _lookup_expected(self, param, index):

        pattern = self._patterns[param]

        slots = index.hour * 4 + index.minute // 15
        doys = index.dayofyear

        if isinstance(pattern, pd.DataFrame):

            available_doys = np.array(pattern.columns)

            values = []

            for slot, doy in zip(slots, doys):

                # Find two nearest days
                diffs = available_doys - doy

                left_mask = diffs <= 0
                right_mask = diffs >= 0

                if left_mask.any():
                    left_doy = available_doys[left_mask].max()
                else:
                    left_doy = available_doys.min()

                if right_mask.any():
                    right_doy = available_doys[right_mask].min()
                else:
                    right_doy = available_doys.max()

                # Ensure slot is in the index
                if slot not in pattern.index:
                    # Use nearest slot if exact slot not found
                    slot = pattern.index[abs(pattern.index - slot).argmin()]
                
                v_left = pattern.loc[slot, left_doy]
                v_right = pattern.loc[slot, right_doy]
                
                # Convert to scalars if they're Series
                if isinstance(v_left, pd.Series):
                    v_left = v_left.iloc[0] if len(v_left) > 0 else np.nan
                if isinstance(v_right, pd.Series):
                    v_right = v_right.iloc[0] if len(v_right) > 0 else np.nan

                if left_doy == right_doy:
                    values.append(float(v_left))
                else:
                    w = (doy - left_doy) / (right_doy - left_doy)
                    values.append(float(v_left * (1 - w) + v_right * w))

            return pd.Series(values, index=index)

        else:

            return pd.Series(
                [pattern.get(slot, pattern.mean()) for slot in slots],
                index=index
            )

    # -------------------------------------------------------
    # Train model
    # -------------------------------------------------------

    def fit(self, df):

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        residual_frames = {}

        for param in df.columns:

            series = df[param].dropna()

            pattern = self._build_pattern(series)

            self._patterns[param] = pattern

            try:
                expected = self._lookup_expected(param, series.index)
                residual_frames[param] = series - expected
            except Exception as e:
                print(f"Warning: Error looking up expected values for {param}: {e}")
                # Skip this parameter if lookup fails
                continue

        residuals_df = pd.DataFrame(residual_frames).dropna()

        if len(residuals_df) == 0:
            raise ValueError("No valid residuals computed - check data quality")

        X_scaled = self.scaler.fit_transform(residuals_df)

        self.model.fit(X_scaled)

        self.feature_names = list(residuals_df.columns)

        self._residual_mean = residuals_df.mean()
        self._residual_std = residuals_df.std().replace(0, 1)

        self.is_fitted = True

        return self

    # -------------------------------------------------------
    # Predict anomalies
    # -------------------------------------------------------

    def predict(self, df):

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        expected_frames = {}
        residual_frames = {}

        for param in self.feature_names:

            expected = self._lookup_expected(param, df.index)

            expected_frames[param] = expected
            residual_frames[param] = df[param] - expected

        expected_df = pd.DataFrame(expected_frames)
        residuals_df = pd.DataFrame(residual_frames)
        
        X_scaled = self.scaler.transform(residuals_df.fillna(0))

        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        return predictions, scores, expected_df, residuals_df

    # -------------------------------------------------------
    # Explain anomaly
    # -------------------------------------------------------

    def explain_anomaly(self, residual_row, actual_row, expected_row, top_n=3):

        results = []

        for param in self.feature_names:

            residual = residual_row[param]

            z = (
                residual - self._residual_mean[param]
            ) / self._residual_std[param]

            results.append({
                "parameter": param,
                "value": round(float(actual_row[param]), 4),
                "expected": round(float(expected_row[param]), 4),
                "deviation": round(float(residual), 4),
                "z_score": round(float(z), 3)
            })

        df = pd.DataFrame(results)

        df["abs_z"] = df["z_score"].abs()

        df = df.sort_values("abs_z", ascending=False)

        return df.drop(columns="abs_z").head(top_n).reset_index(drop=True)

    # -------------------------------------------------------
    # Save / load
    # -------------------------------------------------------

    def save(self, filepath):

        if not self.is_fitted:
            raise ValueError("Model not fitted")

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "patterns": self._patterns,
            "residual_mean": self._residual_mean,
            "residual_std": self._residual_std
        }, filepath)

    def load(self, filepath):

        data = joblib.load(filepath)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._patterns = data["patterns"]
        self._residual_mean = data["residual_mean"]
        self._residual_std = data["residual_std"]

        self.is_fitted = True

        return self