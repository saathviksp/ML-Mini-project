import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'mean_ip', 'std_ip', 'kurt_ip', 'skew_ip',
    'mean_snr', 'std_snr', 'kurt_snr', 'skew_snr'
]
TARGET_NAME = 'target'

# Longer labels used on plot axes and in the demo app.
FEATURE_LABELS = {
    'mean_ip': 'Mean of integrated profile',
    'std_ip': 'Std deviation of integrated profile',
    'kurt_ip': 'Excess kurtosis of integrated profile',
    'skew_ip': 'Skewness of integrated profile',
    'mean_snr': 'Mean of DM-SNR curve',
    'std_snr': 'Std deviation of DM-SNR curve',
    'kurt_snr': 'Excess kurtosis of DM-SNR curve',
    'skew_snr': 'Skewness of DM-SNR curve',
}

RANDOM_STATE = 42


def load_data(filepath='data/HTRU_2.csv'):
    """Loads the HTRU2 dataset and attaches column names.

    The raw CSV has no header row: 8 continuous features followed by the
    binary class label.
    """
    df = pd.read_csv(filepath, header=None)
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    return df


def get_pulsars(df):
    """Returns only the confirmed pulsar rows (target == 1).

    The unsupervised half of this project looks for structure *within* the
    positive class, so the 16,259 non-pulsar rows are dropped here. What is
    left is the 1,639 true pulsars.
    """
    pulsars = df[df[TARGET_NAME] == 1].reset_index(drop=True)
    return pulsars[FEATURE_NAMES]


def scale_features(X):
    """Standardises features to zero mean and unit variance.

    K-means and the SOM both rely on Euclidean distance, and the raw HTRU2
    columns are on very different scales (mean_ip is in the hundreds while
    kurt_ip sits near zero). Without scaling the large-magnitude columns
    would dominate every distance computation.

    Returns the scaled array and the fitted scaler, which the demo app
    reuses so that user input is transformed the same way.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.asarray(X, dtype=np.float64))
    return X_scaled, scaler


def load_pulsar_matrix(filepath='data/HTRU_2.csv'):
    """Convenience wrapper: file path in, scaled pulsar matrix out."""
    df = load_data(filepath)
    X = get_pulsars(df)
    X_scaled, scaler = scale_features(X)
    return X, X_scaled, scaler


def ensure_dir(path):
    """Creates a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path
