"""
Stage 1: turn raw AIS position reports into fixed-length feature windows.

One row of AIS = one position report from one vessel at one moment.
One row of OUTPUT = one 10-report window from one vessel, summarised into features.

Why windows: a quantum circuit takes a fixed-size vector, so each training
example must be a fixed number of values. A single position report is rarely
anomalous on its own -- behaviour over a short stretch is what carries signal.

Run:  .venv/Scripts/python.exe 01_build_windows.py
"""

import numpy as np
import pandas as pd

SRC        = "AIS_2023_01_01.csv"
OUT        = "windows_clean.csv"

# --- filters (identical to the Orange workflow) ---
MAX_SOG      = 40.0    # drops the 102.3 "speed unavailable" sentinel
MIN_MOVING   = 0.5     # knots; below this the vessel is moored, not under way
HEADING_NA   = 511.0   # AIS "heading unavailable"
COG_NA       = 360.0   # AIS "course unavailable"

# --- track / window shape ---
GAP_BREAK_S  = 1800    # >30 min silence = split into separate tracks, don't bridge it
WINDOW       = 10      # reports per window
STRIDE       = 10      # non-overlapping windows (set < WINDOW to overlap)
MIN_REPORTS  = 200     # only keep vessels with rich tracks

USECOLS = ["MMSI","BaseDateTime","LAT","LON","SOG","COG","Heading",
           "VesselType","Length","TransceiverClass"]


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between consecutive points."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing actually travelled between consecutive points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def ang_diff(a, b):
    """Smallest angle between two compass bearings, 0-180."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def load():
    print("reading", SRC, "...")
    # LAT/LON must be float64. float32 holds ~7 significant digits, and a
    # latitude like 34.148670 already uses all of them -- so the DIFFERENCE
    # between two nearby positions, which is exactly what the distance
    # calculation needs, loses most of its precision. Measured error: 0.21% at
    # the typical 272 m spacing, rising to ~5% at 10 m. Found by comparing the
    # pipeline against an independent float64 recomputation (09_independent_check).
    # Costs ~64 MB; buys exact distances and therefore exact implied speeds.
    df = pd.read_csv(SRC, usecols=USECOLS,
                     dtype={"MMSI":"int64","LAT":"float64","LON":"float64",
                            "SOG":"float64","COG":"float64","Heading":"float32",
                            "Length":"float32","TransceiverClass":"string",
                            "VesselType":"float32"},
                     parse_dates=["BaseDateTime"])
    n0 = len(df)

    # Orange filters
    df = df[(df.TransceiverClass == "A") &
            (df.SOG < MAX_SOG) &
            df.LAT.between(-90, 90) &
            df.LON.between(-180, 180)]
    n1 = len(df)

    # under way only -- applied AFTER track splitting logic below via gap breaks
    df = df[df.SOG >= MIN_MOVING]
    n2 = len(df)

    # keep vessels with enough reports to window
    counts = df.MMSI.value_counts()
    keep = counts[counts >= MIN_REPORTS].index
    df = df[df.MMSI.isin(keep)]

    print(f"  raw rows          : {n0:,}")
    print(f"  after AIS filters : {n1:,}")
    print(f"  moving only       : {n2:,}")
    print(f"  vessels >= {MIN_REPORTS} rpts: {df.MMSI.nunique():,}")
    print(f"  rows kept         : {len(df):,}")
    return df.sort_values(["MMSI","BaseDateTime"]).reset_index(drop=True)


def per_step(df):
    """
    Deltas between consecutive reports.

    ORDER MATTERS. Tracks are split FIRST, then deltas are computed within each
    track. Doing it the other way round measures the first report of a track
    against the last report of the previous one -- a different voyage, hours
    later and often hundreds of km away -- which produces impossible implied
    speeds and fake time gaps that look exactly like the anomalies we inject.
    """
    # --- pass 1: find track boundaries using a provisional per-vessel gap ---
    gap = df.groupby("MMSI", sort=False).BaseDateTime.diff().dt.total_seconds()
    new_track = gap.isna() | (gap > GAP_BREAK_S)
    df = df.assign(track_id=new_track.groupby(df.MMSI).cumsum())

    # A "going dark" event is BY DEFINITION longer than the track-split
    # threshold, so it lands on a track boundary and is invisible inside a
    # window. Carry the preceding silence forward as data instead of losing it:
    # every row of a track records how long the vessel was quiet before it began.
    prev_gap = gap.where(new_track)                       # only at boundaries
    df["prev_silence_s"] = (prev_gap.groupby([df.MMSI, df.track_id])
                                    .transform("max")
                                    .fillna(0.0))

    # --- pass 2: all deltas computed WITHIN (MMSI, track_id) ---
    g = df.groupby(["MMSI", "track_id"], sort=False)

    dt   = g.BaseDateTime.diff().dt.total_seconds()
    lat0, lon0 = g.LAT.shift(), g.LON.shift()
    dist = haversine_m(lat0, lon0, df.LAT, df.LON)
    brg  = bearing_deg(lat0, lon0, df.LAT, df.LON)

    df = df.assign(
        dt_s       = dt,
        dist_m     = dist,
        implied_kn = np.where(dt > 0, (dist / dt) * 1.94384, np.nan),
        d_sog      = g.SOG.diff().abs(),
        d_cog      = ang_diff(df.COG, g.COG.shift()),
        bearing    = brg,
    )
    # consistency checks -- a spoofer must keep all of these agreeing, and usually doesn't
    df["spd_mismatch"] = (df.implied_kn - df.SOG).abs()
    df["brg_mismatch"] = np.where(df.COG == COG_NA, np.nan,
                                  ang_diff(df.bearing, df.COG))
    df["crab"]         = np.where((df.Heading == HEADING_NA) | (df.COG == COG_NA), np.nan,
                                  ang_diff(df.COG, df.Heading))
    return df


def windows(df):
    """Collapse each run of WINDOW consecutive reports into one summary row."""
    rows = []
    for (mmsi, tid), t in df.groupby(["MMSI","track_id"], sort=False):
        if len(t) < WINDOW:
            continue
        for s in range(0, len(t) - WINDOW + 1, STRIDE):
            w = t.iloc[s:s+WINDOW]
            rows.append({
                "MMSI": mmsi, "track_id": tid,
                "t_start": w.BaseDateTime.iloc[0],
                "lat_mean": w.LAT.mean(), "lon_mean": w.LON.mean(),   # metadata, NOT features
                "vessel_type": w.VesselType.iloc[0],
                "length": w.Length.iloc[0],
                # --- candidate features ---
                "sog_mean":      w.SOG.mean(),
                "sog_std":       w.SOG.std(ddof=0),
                "sog_max_jump":  w.d_sog.max(),
                "cog_turn_max":  w.d_cog.max(),
                "cog_turn_mean": w.d_cog.mean(),
                "dt_max":        w.dt_s.max(),
                "dt_mean":       w.dt_s.mean(),
                "dist_total_m":  w.dist_m.sum(),
                "prev_silence_s": w.prev_silence_s.iloc[0],   # track-level going-dark signal
                "spd_mismatch_max":  w.spd_mismatch.max(),
                "spd_mismatch_mean": w.spd_mismatch.mean(),
                "brg_mismatch_mean": w.brg_mismatch.mean(),
                "crab_mean":         w.crab.mean(),
                "heading_valid_frac": float((w.Heading != HEADING_NA).mean()),
            })
    return pd.DataFrame(rows)


MAX_IMPLIED_KN = 60.0   # no commercial vessel exceeds this; anything above is a data error


if __name__ == "__main__":
    df = load()
    df = per_step(df)
    print("\nbuilding windows ...")
    W = windows(df)

    # Drop windows containing physically impossible values. These are data errors
    # -- most often two different vessels transmitting the same MMSI, which makes
    # their interleaved positions look like teleportation. We are building the
    # CLEAN baseline here, so real corruption must go: otherwise it sits in the
    # "normal" class and silently mislabels the training data.
    n_before = len(W)
    bad = (W.spd_mismatch_max > MAX_IMPLIED_KN) | (W.dt_max > GAP_BREAK_S)
    W = W[~bad].reset_index(drop=True)
    print(f"  dropped {n_before - len(W):,} windows with impossible values "
          f"({100*(n_before-len(W))/n_before:.1f}%)")

    W.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}")
    print(f"  windows : {len(W):,}")
    print(f"  vessels : {W.MMSI.nunique():,}")
    print(f"  tracks  : {len(W.groupby(['MMSI','track_id'])):,}")
    feats = [c for c in W.columns if c not in
             ("MMSI","track_id","t_start","lat_mean","lon_mean","vessel_type","length")]
    print(f"\nfeature summary ({len(feats)} candidates):")
    print(W[feats].describe().T[["mean","std","min","max"]].round(2).to_string())
    print("\nmissing per feature:")
    print(W[feats].isna().mean().mul(100).round(1).to_string())
