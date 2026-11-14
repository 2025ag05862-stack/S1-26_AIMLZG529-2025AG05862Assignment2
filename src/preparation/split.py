"""Time-based train/test split: past -> train, future -> test (no leakage from the future)."""
import pandas as pd


def time_split(x: pd.DataFrame, test_fraction: float):
    cutoff = x.timestamp.quantile(1 - test_fraction)
    train = x[x.timestamp <= cutoff].reset_index(drop=True)
    test = x[x.timestamp > cutoff].reset_index(drop=True)
    return train, test, cutoff
