"""
Preprocessing audit: every filter, in order, with rows and vessels remaining.

This is the data-flow table for the paper's methods section. Each row is one
filter and shows exactly what it cost.

Run:  .venv/Scripts/python.exe 00_preprocessing_audit.py
"""

import pandas as pd

SRC = "data/raw/AIS_2023_01_01.csv"
USECOLS = ["MMSI","BaseDateTime","LAT","LON","SOG","COG","Heading",
           "VesselType","Length","TransceiverClass"]

steps = []


def note(label, df, reason):
    steps.append({
        "step": label,
        "rows": len(df),
        "vessels": df.MMSI.nunique(),
        "why": reason,
    })


print("loading ...")
df = pd.read_csv(SRC, usecols=USECOLS,
                 dtype={"MMSI":"int64","LAT":"float32","LON":"float32",
                        "SOG":"float32","COG":"float32","Heading":"float32",
                        "Length":"float32","TransceiverClass":"string",
                        "VesselType":"float32"},
                 parse_dates=["BaseDateTime"])
note("0. raw file", df, "as downloaded from NOAA")

df = df[df.TransceiverClass == "A"]
note("1. Class A only", df, "Class B reports every 30s-3min vs A's 2-10s; mixing breaks time-gap features")

df = df[df.SOG < 40]
note("2. SOG < 40 kn", df, "removes the 102.3 'speed unavailable' sentinel and impossible speeds")

df = df[df.LAT.between(-90, 90) & df.LON.between(-180, 180)]
note("3. valid coordinates", df, "removes transmission/parsing corruption")

df = df[df.SOG >= 0.5]
note("4. under way only (SOG >= 0.5)", df, "REMOVES MOORED/STATIONARY VESSELS - 68% of all reports")

counts = df.MMSI.value_counts()
df = df[df.MMSI.isin(counts[counts >= 200].index)]
note("5. vessels with >= 200 reports", df, "ensures each vessel yields ~20 windows; prevents one vessel dominating")

T = pd.DataFrame(steps)
T["rows_lost"] = T.rows.diff().fillna(0).astype(int)
T["pct_of_raw"] = (100 * T.rows / T.rows.iloc[0]).round(1)

print("\n" + "=" * 100)
print("PREPROCESSING WATERFALL")
print("=" * 100)
print(T[["step","rows","rows_lost","pct_of_raw","vessels"]].to_string(index=False))
print("\nreasons:")
for s in steps:
    print(f"  {s['step']:32s} {s['why']}")

print("\n" + "=" * 100)
print("SOG DISTRIBUTION BEFORE vs AFTER the stationary filter")
print("=" * 100)
raw = pd.read_csv(SRC, usecols=["SOG","TransceiverClass"],
                  dtype={"SOG":"float32","TransceiverClass":"string"})
raw = raw[(raw.TransceiverClass == "A") & (raw.SOG < 40)]
print(f"  before:  n={len(raw):,}  mean={raw.SOG.mean():.2f} kn  "
      f"median={raw.SOG.median():.2f}  zero-or-near={100*(raw.SOG<0.5).mean():.1f}%")
print(f"  after :  n={len(df):,}  mean={df.SOG.mean():.2f} kn  "
      f"median={df.SOG.median():.2f}  zero-or-near={100*(df.SOG<0.5).mean():.1f}%")
