"""
================================================================================
COMPLETE DATA PIPELINE — raw AIS to a labelled, split, encoding-ready dataset
================================================================================

Everything the study does to the data, in one runnable file, in order.
Read it top to bottom to understand the whole preparation stage.

    8,156,509 raw AIS reports (one day, US coastal waters)
            |  five filters
    1,256,434 usable reports
            |  split into voyages at 30-min silences
        5,928 tracks
            |  compute per-step differences, aggregate 10 reports -> 1 window
      122,371 windows
            |  inject five documented attack types
      114,972 windows, 3.5% anomalous
            |  log-transform, scale to [0, pi], stratified split by vessel
    train / validation / test

EVERY CONSTANT IS DERIVED. Each is tagged with what it was derived FROM.
Reference measurements taken from 1,256,434 unmodified reports:

    distance between consecutive reports : median 272 m, p90 551 m, p99 1,285 m
    gap between consecutive reports      : median  70 s, p99 491 s, p99.9 1,409 s
    window-max gap (max of 10 intervals) : median  79 s, p95 580 s, p99 1,318 s
    speed / position mismatch            : median 0.11 kn, p99 3.20, p99.9 11.27
    speed reported while moored          : median 0.00 kn, p90 0.10
    speed reported while at anchor       : median 0.00 kn, p95 0.40

Run:  .venv/Scripts/python.exe PIPELINE.py
"""

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION — every value derived, with its justification
# ==============================================================================

SRC = "AIS_2023_01_01.csv"

# ---- row filters -------------------------------------------------------------
MAX_SOG    = 40.0   # DERIVED: AIS encodes "speed unavailable" as 102.3. 40 kn also
                    # exceeds any commercial vessel's service speed.
MIN_MOVING = 0.5    # DERIVED: above the p95 of vessels declaring themselves at
                    # anchor (0.40) and the p90 of moored vessels (0.10). Those
                    # small readings are GPS jitter at a berth, not motion.
HEADING_NA = 511.0  # AIS sentinel: heading unavailable
COG_NA     = 360.0  # AIS sentinel: course unavailable

# ---- track construction ------------------------------------------------------
GAP_BREAK_S = 1800  # DERIVED: 0.000% of legitimate Class A intervals reach 1800 s.
                    # The threshold sits entirely outside normal behaviour.
MIN_REPORTS = 200   # REASONED: at 10 reports per window this gives ~20 windows per
                    # vessel, so no single ship's quirks dominate a class.
MAX_IMPLIED_KN = 60.0  # DERIVED: no vessel exceeds this. In practice these are
                       # duplicate MMSIs -- two ships sharing an identifier, whose
                       # interleaved positions look like teleportation.

# ---- windows -----------------------------------------------------------------
WINDOW = 10  # REASONED: at the median 70 s interval, 10 reports span ~12 minutes --
             # long enough to contain a manoeuvre, short enough that an anomaly is
             # not diluted. Swept at 5/10/20; 10 maximises no single attack.
STRIDE = 10  # DERIVED: equal to WINDOW means non-overlapping. Overlapping windows
             # share reports, and two of them on opposite sides of a split is
             # leakage that vessel-level splitting would not catch.

# ---- injection ---------------------------------------------------------------
ANOMALY_FRAC = 0.45   # DERIVED from what the experiment needs: the label-scarcity
                      # curve trains on up to 500 anomalies across 5 seeds, so
                      # ~2,500 anomalous training windows is the requirement.
N_TRACKS     = 5900   # essentially every available track

# Severity ranges. Each bound anchored to a measured property of real data.
DRIFT_M      = (100,  3000)   # lower is BELOW the median single-step movement
                              # (272 m), so the vessel's own normal motion exceeds
                              # the whole injected drift -- undetectable by design.
JUMP_M       = (500, 20000)   # at the median 79 s interval: 500 m implies 12.3 kn
                              # (plausible, near the p90 of normal movement);
                              # 20 km implies 492 kn (impossible).
GAP_S        = (500,  1700)   # lower is just above the p99 of natural gaps (491 s);
                              # upper is just under GAP_BREAK_S, or the pipeline
                              # splits the track and the anomaly ceases to exist.
SPEED_LIE    = (1.8,   5.0)   # MULTIPLIER. On a 7.7 kn vessel, x1.8 creates a
                              # 6.2 kn mismatch (above natural p99 of 3.20);
                              # x5.0 creates 30.8 kn (far above p99.9 of 11.27).
DARK_S       = (3600, 21600)  # must EXCEED GAP_BREAK_S to register as a track-level
                              # blackout. Upper bound 6 h is where a vessel could
                              # plausibly have left receiver coverage legitimately.

# Attacks yield very different window counts because their masks differ in width
# (speed_lie corrupts hundreds of rows, jump corrupts three). Even allocation gave
# a 128:1 imbalance, so tracks are allocated INVERSELY to measured yield.
YIELD_WEIGHT = {"speed_lie": 1.0, "drift": 2.5, "jump": 12.0, "gap": 16.0, "dark": 22.0}
ATTACKS = list(YIELD_WEIGHT)

# ---- splits ------------------------------------------------------------------
TEST_FRAC = 0.25
VAL_FRAC  = 0.15   # leaves 60% for training

RNG = np.random.default_rng(42)

USECOLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading",
           "VesselType", "Length", "TransceiverClass"]

# The 7 engineered features plus a missingness flag = 8 model inputs
CORE = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
        "spd_mismatch_max", "crab_mean", "prev_silence_s"]
# These span orders of magnitude. Angle encoding maps each feature to a rotation
# between 0 and pi; without a log transform 99% of values would map to nearly the
# same angle and that qubit would carry almost no information.
LOGGED = ["sog_max_jump", "cog_turn_max", "dt_max", "spd_mismatch_max",
          "crab_mean", "prev_silence_s"]


# ==============================================================================
# GEOMETRY
# ==============================================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres.

    Haversine rather than the spherical law of cosines: the latter takes arccos
    of a value extremely close to 1 for short distances and loses precision there.
    """
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing actually travelled between two positions."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def ang_diff(a, b):
    """Smallest angle between two compass bearings, 0-180 degrees."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def m_to_deg(m, lat):
    """Metres to degrees of latitude / longitude at a given latitude."""
    return m / 111_320.0, m / (111_320.0 * max(np.cos(np.radians(lat)), 0.01))


# ==============================================================================
# STAGE 1 — LOAD AND FILTER
# ==============================================================================

def load():
    """8,156,509 raw reports -> 1,256,434 usable ones."""
    print("STAGE 1: load and filter")

    # LAT/LON MUST be float64. float32 holds ~7 significant digits, and a latitude
    # like 34.148670 already uses all of them -- so the DIFFERENCE between two
    # nearby positions, which is exactly what the distance calculation needs,
    # loses most of its precision. Measured error: 0.21% at the typical 272 m
    # spacing, rising to ~5% at 10 m. This silently degraded spd_mismatch_max,
    # the strongest feature, until an independent recomputation caught it.
    df = pd.read_csv(SRC, usecols=USECOLS,
                     dtype={"MMSI": "int64", "LAT": "float64", "LON": "float64",
                            "SOG": "float64", "COG": "float64",
                            "Heading": "float32", "Length": "float32",
                            "VesselType": "float32", "TransceiverClass": "string"},
                     parse_dates=["BaseDateTime"])
    n0 = len(df)

    df = df[df.TransceiverClass == "A"]            # scoping, not cleaning
    df = df[df.SOG < MAX_SOG]                      # drops the 102.3 sentinel
    df = df[df.LAT.between(-90, 90) & df.LON.between(-180, 180)]
    n1 = len(df)

    df = df[df.SOG >= MIN_MOVING]                  # drops moored vessels
    n2 = len(df)

    counts = df.MMSI.value_counts()
    df = df[df.MMSI.isin(counts[counts >= MIN_REPORTS].index)]

    print(f"  raw                    {n0:>10,}")
    print(f"  after AIS filters      {n1:>10,}")
    print(f"  under way only         {n2:>10,}   (-{n1-n2:,} moored)")
    print(f"  vessels >= {MIN_REPORTS} reports {len(df):>10,}   ({df.MMSI.nunique():,} vessels)")
    return df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)


# ==============================================================================
# STAGE 2 — TRACKS AND PER-STEP FEATURES
# ==============================================================================

def per_step(df):
    """Split voyages, then compute differences between consecutive reports.

    ORDER MATTERS AND IS NOT NEGOTIABLE. Splitting must happen FIRST. Computing
    differences before splitting compares the first report of one voyage against
    the last report of the previous one -- hours later and hundreds of km away.
    Doing it the wrong way round produced implied speeds of 258,249 knots and
    displacements of 2,225 km inside a 10-report window.
    """
    # --- pass 1: find voyage boundaries ---
    gap = df.groupby("MMSI", sort=False).BaseDateTime.diff().dt.total_seconds()
    new_track = gap.isna() | (gap > GAP_BREAK_S)
    df = df.assign(track_id=new_track.groupby(df.MMSI).cumsum())

    # A blackout is BY DEFINITION longer than the split threshold, so it lands on
    # a track boundary and is invisible inside any window. Carry the preceding
    # silence forward as data instead of losing it at the seam.
    prev_gap = gap.where(new_track)
    df["prev_silence_s"] = (prev_gap.groupby([df.MMSI, df.track_id])
                                    .transform("max").fillna(0.0))

    # --- pass 2: all deltas computed WITHIN (MMSI, track_id) ---
    g = df.groupby(["MMSI", "track_id"], sort=False)
    dt = g.BaseDateTime.diff().dt.total_seconds()
    lat0, lon0 = g.LAT.shift(), g.LON.shift()
    dist = haversine_m(lat0, lon0, df.LAT, df.LON)

    df = df.assign(
        dt_s       = dt,
        dist_m     = dist,
        # what the POSITIONS say the speed was, in knots
        implied_kn = np.where(dt > 0, (dist / dt) * 1.94384, np.nan),
        d_sog      = g.SOG.diff().abs(),
        d_cog      = ang_diff(df.COG, g.COG.shift()),
        bearing    = bearing_deg(lat0, lon0, df.LAT, df.LON),
    )

    # THE KEY FEATURE. A spoofer fabricating positions must keep the reported
    # speed consistent with them, and usually doesn't. This needs no idea what
    # normal looks like -- only that the vessel contradicts itself.
    df["spd_mismatch"] = (df.implied_kn - df.SOG).abs()
    # course vs heading: legitimate in a current ("crabbing"), but fabricated
    # data often forgets to keep them consistent
    df["crab"] = np.where((df.Heading == HEADING_NA) | (df.COG == COG_NA), np.nan,
                          ang_diff(df.COG, df.Heading))
    return df


def windows(df):
    """Collapse each run of WINDOW consecutive reports into one summary row."""
    rows = []
    for (mmsi, tid), t in df.groupby(["MMSI", "track_id"], sort=False):
        if len(t) < WINDOW:
            continue
        for s in range(0, len(t) - WINDOW + 1, STRIDE):
            w = t.iloc[s:s+WINDOW]
            rows.append({
                "MMSI": mmsi, "track_id": tid,
                "t_start": w.BaseDateTime.iloc[0],
                # metadata, never model inputs -- raw position would let the model
                # learn "anomalies happen near Los Angeles"
                "lat_mean": w.LAT.mean(), "lon_mean": w.LON.mean(),
                "vessel_type": w.VesselType.iloc[0], "length": w.Length.iloc[0],
                # --- the features ---
                "sog_mean":         w.SOG.mean(),
                "sog_max_jump":     w.d_sog.max(),
                "cog_turn_max":     w.d_cog.max(),
                "dt_max":           w.dt_s.max(),
                "spd_mismatch_max": w.spd_mismatch.max(),
                "crab_mean":        w.crab.mean(),
                "prev_silence_s":   w.prev_silence_s.iloc[0],
                # carried through so window labels can be assigned later
                "anom_row": w.anom_row.any() if "anom_row" in w else False,
                "attack":   w.attack.iloc[0] if "attack" in w else "none",
                "severity": w.severity.iloc[0] if "severity" in w else 0.0,
            })
    return pd.DataFrame(rows)


# ==============================================================================
# STAGE 3 — THE FIVE ATTACKS
# ==============================================================================
# Each returns (modified_track, boolean mask of AFFECTED ROWS).
#
# The mask matters enormously. An earlier version injected an anomaly into a
# TRACK and labelled every window of that track anomalous. A track yields ~20
# windows; an anomaly touches 1-3. So 85-90% of "anomalous" windows contained no
# anomaly at all. The mask records exactly which rows changed, and a window is
# labelled anomalous only if it CONTAINS one.

def a_drift(t, sev):
    """Positions diverge gradually along a random bearing. Reported speed
    untouched, so implied speed slowly stops matching it. This is what
    sophisticated spoofing looks like -- the track stays plausible throughout."""
    n = len(t)
    ramp = np.linspace(0, 1, n) ** 1.5          # slow start, accelerating
    br = RNG.uniform(0, 2*np.pi)
    dlat, dlon = m_to_deg(sev, t.LAT.mean())
    t["LAT"] = t.LAT.values + ramp * dlat * np.sin(br)
    t["LON"] = t.LON.values + ramp * dlon * np.cos(br)
    mask = np.zeros(n, bool); mask[n//2:] = True   # only detectable once accumulated
    return t, mask


def a_jump(t, sev):
    """One instantaneous displacement; the track continues from the new position.
    Detected because the implied speed across the discontinuity is impossible."""
    n = len(t)
    k = int(RNG.integers(n//4, max(n//4+1, 3*n//4)))
    br = RNG.uniform(0, 2*np.pi)
    dlat, dlon = m_to_deg(sev, t.LAT.mean())
    t.iloc[k:, t.columns.get_loc("LAT")] = t.LAT.values[k:] + dlat*np.sin(br)
    t.iloc[k:, t.columns.get_loc("LON")] = t.LON.values[k:] + dlon*np.cos(br)
    mask = np.zeros(n, bool); mask[max(0,k-1):min(n,k+2)] = True
    return t, mask


def a_gap(t, sev):
    """Delete reports so the vessel falls briefly silent, WITHIN one voyage."""
    n = len(t)
    if n < 8:
        return t, np.zeros(n, bool)
    k = int(RNG.integers(2, n-4))
    dt = t.BaseDateTime.diff().dt.total_seconds().fillna(60).values
    drop, acc = [], 0.0
    for i in range(k, n-2):
        acc += dt[i]; drop.append(i)
        if acc >= sev: break
    keep = np.ones(n, bool); keep[drop] = False
    t2 = t[keep].reset_index(drop=True)
    mask = np.zeros(len(t2), bool)
    j = min(k, len(t2)-1)
    mask[max(0, j-1):min(len(t2), j+1)] = True   # rows straddling the silence
    return t2, mask


def a_speed_lie(t, sev):
    """Positions untouched; the reported SOG field is falsified.

    The multiplier is chosen from the headroom actually available in each
    direction. An earlier version added or subtracted a fixed number of knots and
    clipped at 0.1 -- so on a 7-knot vessel a negative offset bottomed out while a
    positive one did not, pinning 30% of windows against a bound and teaching the
    model that "slow = anomalous". That was an artefact of the code, not spoofing.
    """
    FLOOR, CEIL = 0.5, 39.0
    n = len(t)
    k = int(RNG.integers(0, max(n//2, 1)))
    s = t.SOG.values[k:]
    up   = CEIL / max(s.max(), 0.1)
    down = max(s.min(), 0.1) / FLOOR
    dirs = []
    if up   > 1.2: dirs.append(("up",   min(sev, up)))
    if down > 1.2: dirs.append(("down", min(sev, down)))
    if not dirs:
        return t, np.zeros(n, bool)
    which, mag = dirs[int(RNG.integers(len(dirs)))]
    factor = mag if which == "up" else 1.0 / mag
    t.iloc[k:, t.columns.get_loc("SOG")] = np.clip(s * factor, FLOOR, CEIL)
    mask = np.zeros(n, bool); mask[k:] = True
    return t, mask


def a_dark(t, sev):
    """Transponder switched off for hours. Deleting a stretch longer than
    GAP_BREAK_S splits the track, so the second half carries a large preceding
    silence -- which is what prev_silence_s exists to detect. This is a
    TRACK-level event, invisible inside any single window."""
    n = len(t)
    if n < 3 * WINDOW:
        return t, np.zeros(n, bool)
    dt = t.BaseDateTime.diff().dt.total_seconds().fillna(60).values
    total = float(np.nansum(dt))
    if total < 2 * GAP_BREAK_S:
        return t, np.zeros(n, bool)
    target = min(sev, 0.40 * total)      # cap so the voyage survives the deletion
    if target < GAP_BREAK_S * 1.1:
        return t, np.zeros(n, bool)
    k = int(RNG.integers(WINDOW, max(WINDOW + 1, n - 2*WINDOW)))
    drop, acc = [], 0.0
    for i in range(k, n - WINDOW):
        acc += dt[i]; drop.append(i)
        if acc >= target: break
    if acc < GAP_BREAK_S:
        return t, np.zeros(n, bool)
    keep = np.ones(n, bool); keep[drop] = False
    t2 = t[keep].reset_index(drop=True)
    mask = np.zeros(len(t2), bool)
    j = min(k, len(t2) - 1)
    mask[j:min(len(t2), j + WINDOW)] = True
    return t2, mask


APPLY = {"drift": (a_drift, DRIFT_M), "jump": (a_jump, JUMP_M),
         "gap": (a_gap, GAP_S), "speed_lie": (a_speed_lie, SPEED_LIE),
         "dark": (a_dark, DARK_S)}


def inject(df):
    """Attack a share of tracks, allocated inversely to measured window yield."""
    print("\nSTAGE 3: inject anomalies")
    keys = df.groupby(["MMSI", "track_id"]).size()
    keys = keys[keys >= WINDOW].index.to_list()
    RNG.shuffle(keys)
    keys = keys[:N_TRACKS]

    n_anom = int(len(keys) * ANOMALY_FRAC)
    w = np.array([YIELD_WEIGHT[a] for a in ATTACKS], float)
    quota = np.floor(n_anom * w / w.sum()).astype(int)
    assign, pos = {}, 0
    for a, q in zip(ATTACKS, quota):
        for k in keys[pos:pos+q]:
            assign[tuple(k)] = a
        pos += q
    print(f"  {len(keys):,} tracks, {len(assign):,} attacked")
    print("  allocation: " + "  ".join(f"{a}={q}" for a, q in zip(ATTACKS, quota)))

    groups = {k: g for k, g in df.groupby(["MMSI", "track_id"], sort=False)}
    out = []
    for key in keys:
        t = groups[key].copy().reset_index(drop=True)
        atk, sev, mask = "none", 0.0, np.zeros(len(t), bool)
        if tuple(key) in assign:
            atk = assign[tuple(key)]
            fn, rng_ = APPLY[atk]
            sev = float(RNG.uniform(*rng_))
            t, mask = fn(t, sev)
            if not mask.any():           # attack could not be applied to this track
                atk, sev = "none", 0.0
        t["anom_row"], t["attack"], t["severity"] = mask, atk, sev
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ==============================================================================
# STAGE 4 — SCALE AND SPLIT
# ==============================================================================

def split_and_scale(W):
    print("\nSTAGE 4: scale and split")

    # 19% of windows have no valid heading (AIS sends 511 when the sensor is not
    # reporting). Impute the median and flag it -- the flag may itself be
    # informative, since a vessel that stops reporting heading is doing something.
    W = W.copy()
    W["crab_valid"] = (~W.crab_mean.isna()).astype(int)
    W["crab_mean"]  = W.crab_mean.fillna(W.crab_mean.median())

    # STRATIFIED split BY VESSEL.
    #   By vessel, because ten consecutive reports from one ship and the next ten
    #   minutes later are near-identical. Split randomly by ROW and those twins
    #   land on opposite sides -- the model recognises the ship, not the behaviour.
    #   Stratified, because vessels differ in how many anomalies they carry, and a
    #   plain random split left validation at 2.6% against test at 4.0%.
    #   Stratified on BOTH anomaly count and dominant attack type. Balancing
    #   count alone still let rare attacks cluster into one split -- validation
    #   ended up with a 6.3x imbalance between attack types, leaving as few as 29
    #   windows of the rarest one.
    per_v = (W.groupby("MMSI").agg(n=("label", "size"), k=("label", "sum"))
               .reset_index())
    dom = (W[W.label == 1].groupby("MMSI").attack
             .agg(lambda s: s.value_counts().index[0]))
    per_v["dom"] = per_v.MMSI.map(dom).fillna("none")
    per_v["cnt"] = per_v.k.map(lambda k: 0 if k == 0 else 1 if k <= 2 else 2 if k <= 8 else 3)
    per_v["bin"] = per_v.dom + "_" + per_v.cnt.astype(str)

    assign_split = {}
    for _, grp in per_v.groupby("bin"):
        v = grp.MMSI.to_numpy().copy()
        RNG.shuffle(v)
        n_te = int(round(len(v) * TEST_FRAC))
        n_va = int(round(len(v) * VAL_FRAC))
        for m in v[:n_te]:            assign_split[m] = "test"
        for m in v[n_te:n_te+n_va]:   assign_split[m] = "val"
        for m in v[n_te+n_va:]:       assign_split[m] = "train"

    col = W.MMSI.map(assign_split)
    tr, va, te = (W[col == s].reset_index(drop=True) for s in ("train", "val", "test"))

    feats = CORE + ["crab_valid"]
    for d in (tr, va, te):
        for c in LOGGED:
            d[c] = np.log10(d[c].clip(lower=0) + 1e-2)

    # Scale to [0, pi] for angle encoding, using TRAIN statistics ONLY.
    # Computing min/max across all data would leak test information into training.
    lo, hi = tr[feats].min(), tr[feats].max()
    span = (hi - lo).replace(0, 1)
    for d in (tr, va, te):
        d[feats] = ((d[feats] - lo) / span).clip(0, 1) * np.pi

    tr.to_csv("dataset_train.csv", index=False)
    va.to_csv("dataset_val.csv", index=False)
    te.to_csv("dataset_test.csv", index=False)

    for name, d in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s} {len(d):>7,} windows  {d.MMSI.nunique():>5,} vessels  "
              f"{int(d.label.sum()):>5,} anomalous ({100*d.label.mean():.2f}%)")
    for a, b_, nm in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        ov = set(a.MMSI) & set(b_.MMSI)
        print(f"  vessel overlap {nm:11s} {len(ov)}  {'OK' if not ov else 'LEAKAGE'}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    df = load()
    print("\nSTAGE 2: build tracks and per-step features")
    df = per_step(df)
    print(f"  {len(df.groupby(['MMSI','track_id'])):,} voyage tracks")

    df = df.astype({"LAT": "float64", "LON": "float64", "SOG": "float64"})
    D = inject(df)

    # Recompute everything on the MODIFIED tracks -- an injected drift must show
    # up in implied speed, bearing and turn rate, and those are derived quantities.
    print("\n  recomputing features on modified tracks ...")
    D = D.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    D = per_step(D.drop(columns=[c for c in
        ["dt_s","dist_m","implied_kn","d_sog","d_cog","bearing","spd_mismatch",
         "crab","track_id","prev_silence_s"] if c in D.columns]))

    W = windows(D)
    W["label"] = W.anom_row.astype(int)
    W.loc[W.label == 0, ["attack", "severity"]] = ["none", 0.0]
    W = W.drop(columns=["anom_row"])

    # Reapply the sanity filter to NORMAL windows only. Anomalous windows
    # legitimately produce extreme values; corrupt normal ones would sit in the
    # negative class and blur the very contrast the study measures.
    n0 = len(W)
    W = W[~((W.spd_mismatch_max > MAX_IMPLIED_KN) & (W.label == 0))].reset_index(drop=True)
    print(f"  {len(W):,} windows  ({n0-len(W):,} corrupt normal windows dropped)")
    print(f"  {int(W.label.sum()):,} anomalous ({100*W.label.mean():.1f}%)")
    print("  by attack: " + str(W[W.label == 1].attack.value_counts().to_dict()))

    W.to_csv("windows_labelled.csv", index=False)
    split_and_scale(W)
    print("\nDone. Verify with 07_validate_dataset.py and 09_independent_check.py")


if __name__ == "__main__":
    main()
