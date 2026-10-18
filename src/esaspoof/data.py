"""
Loading and preprocessing of ESA-ADB Mission1 telemetry.

Reads the raw per-channel pickles, resamples them onto a uniform time grid,
and builds a binary anomaly mask from labels.csv.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ESA-Mission1"
PROCESSED = ROOT / "data" / "processed"


def load_channel(name):
    # each zip holds one pickled DataFrame with a datetime index
    return pd.read_pickle(RAW / "channels" / f"{name}.zip")


def load_group(names, freq="10min"):
    # mean per time bin -> all channels share the same uniform index
    series = [load_channel(n).resample(freq).mean() for n in names]
    return pd.concat(series, axis=1)


def anomaly_mask(index, names, freq="10min"):
    labels = pd.read_csv(RAW / "labels.csv", parse_dates=["StartTime", "EndTime"])
    labels = labels[labels["Channel"].isin(names)]

    # keep real anomalies only, drop "Rare Event" entries
    types = pd.read_csv(RAW / "anomaly_types.csv")
    anomaly_ids = types.loc[types["Category"] == "Anomaly", "ID"]
    labels = labels[labels["ID"].isin(anomaly_ids)]

    mask = pd.DataFrame(0, index=index, columns=names, dtype="int8")
    for _, r in labels.iterrows():
        # labels are UTC-aware, the telemetry index is naive -> strip tz
        # floor the start so short anomalies still hit at least one bin
        start = r["StartTime"].tz_localize(None).floor(freq)
        end = r["EndTime"].tz_localize(None)
        mask.loc[start:end, r["Channel"]] = 1
    return mask

SPLITS = {
    "train": ("2000-01-01", "2007-12-31"),
    "val":   ("2008-01-01", "2009-12-31"),
    "test":  ("2010-01-01", "2013-12-31"),
}


def load_processed():
    df = pd.read_parquet(PROCESSED / "group8_10min.parquet")
    mask = pd.read_parquet(PROCESSED / "group8_mask.parquet")
    # only ~0.005% missing -> carry the last value forward
    df = df.ffill().bfill()
    return df, mask


def make_splits(df, mask):
    # z-score with nominal train data only, so anomalies and val/test don't leak in
    train = df.loc[slice(*SPLITS["train"])]
    nominal = train[mask.loc[train.index].max(axis=1) == 0]
    mu, sigma = nominal.mean(), nominal.std()

    splits = {}
    for name, (start, end) in SPLITS.items():
        splits[name] = ((df.loc[start:end] - mu) / sigma, mask.loc[start:end])
    return splits, (mu, sigma)