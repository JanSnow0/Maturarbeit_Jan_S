from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

CH_TZ = "Europe/Zurich"

# ── TZ helpers ───────────────────────────────────────────────────────────────
def to_ch(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce", utc=True).tz_convert(CH_TZ)
        return idx
    if idx.tz is None:
        return idx.tz_localize(CH_TZ)
    return idx.tz_convert(CH_TZ)

def ny_window_ch(day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(day, tz=CH_TZ)
    return (
        d.replace(hour=15, minute=30, second=0, microsecond=0),
        d.replace(hour=17, minute=0, second=0, microsecond=0),
    )

# ── CSV loader (robust to mixed tz strings) ─────────────────────────────────
def load_symbol_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    # detect a timestamp column
    for tcol in ["timestamp","Timestamp","datetime","time","Time","Date","date"]:
        if tcol in df.columns:
            ts = pd.to_datetime(df[tcol], errors="coerce", utc=True)
            break
    else:
        raise ValueError(f"No timestamp column found in {path.name}")

    if ts.isna().all():
        raise ValueError(f"Could not parse datetimes in {path.name}")

    ts = ts.dt.tz_convert(CH_TZ)
    df = df.drop(columns=[tcol])

    # coerce numerics
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = ts
    df.sort_index(inplace=True)
    return df

# ── Column helpers ──────────────────────────────────────────────────────────
def _pick_cols(df: pd.DataFrame, candidates: list[list[str]]) -> list[str] | None:
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    return None

def _extract_1m(df: pd.DataFrame) -> pd.DataFrame:
    cols = _pick_cols(df, [
        ["open","high","low","close"],
        ["open_1m","high_1m","low_1m","close_1m"],
        ["open_1m_live","high_1m_live","low_1m_live","close_1m_live"],
        ["o","h","l","c"],
    ])
    if not cols:
        raise ValueError("1m OHLC columns not found")
    out = df[cols].copy()
    out.columns = ["open","high","low","close"]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.index = to_ch(df.index)
    return out

# ── Finished bars (prefer precomputed; else fallback) ────────────────────────
def finished_5m(df: pd.DataFrame) -> pd.DataFrame:
    cols = _pick_cols(df, [
        ["open_5m","high_5m","low_5m","close_5m"],
        ["open_5min","high_5min","low_5min","close_5min"],
        ["open_5m_live","high_5m_live","low_5m_live","close_5m_live"],
    ])
    if cols:
        out = df[cols].copy()
        out.columns = ["open","high","low","close"]
        out.index = to_ch(df.index)
        return out.dropna().sort_index()
    m1 = _extract_1m(df)
    return pd.DataFrame({
        "open":  m1["open"].resample("5T", label="right", closed="right").first(),
        "high":  m1["high"].resample("5T", label="right", closed="right").max(),
        "low":   m1["low"].resample("5T", label="right", closed="right").min(),
        "close": m1["close"].resample("5T", label="right", closed="right").last(),
    }).dropna().sort_index()

def finished_1h(df: pd.DataFrame) -> pd.DataFrame:
    cols = _pick_cols(df, [
        ["open_1h","high_1h","low_1h","close_1h"],
        ["open_60m","high_60m","low_60m","close_60m"],
        ["open_1h_live","high_1h_live","low_1h_live","close_1h_live"],
    ])
    if cols:
        out = df[cols].copy()
        out.columns = ["open","high","low","close"]
        out.index = to_ch(df.index)
        return out.dropna().sort_index()
    m1 = _extract_1m(df)
    return pd.DataFrame({
        "open":  m1["open"].resample("60T", label="right", closed="right").first(),
        "high":  m1["high"].resample("60T", label="right", closed="right").max(),
        "low":   m1["low"].resample("60T", label="right", closed="right").min(),
        "close": m1["close"].resample("60T", label="right", closed="right").last(),
    }).dropna().sort_index()

# ── Prev hour & sessions ────────────────────────────────────────────────────
def attach_prev_hour_levels(df5: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    """
    For each 5m bar, attach the previous COMPLETED 1h high/low.
    Robust even if indices have no names (explicit ts/h_ts columns).
    """
    f5 = df5.copy(); f5.index = to_ch(f5.index)
    h1 = df1h.copy(); h1.index = to_ch(h1.index)

    left = f5.reset_index()
    right = h1.reset_index()

    left = left.rename(columns={left.columns[0]: "ts"})
    right = right.rename(columns={right.columns[0]: "h_ts"})

    left = left.sort_values("ts")
    right = right.sort_values("h_ts")

    asof = pd.merge_asof(
        left, right,
        left_on="ts", right_on="h_ts",
        direction="backward", allow_exact_matches=False,
        suffixes=("_5m","_1h")
    )

    out = f5.copy()
    out["prev_hour_high"] = asof["high_1h"].values
    out["prev_hour_low"]  = asof["low_1h"].values
    return out

def _session_window(day: str, start_hm: tuple[int,int], end_hm: tuple[int,int]):
    d = pd.Timestamp(day, tz=CH_TZ)
    s = d.replace(hour=start_hm[0], minute=start_hm[1], second=0, microsecond=0)
    e = d.replace(hour=end_hm[0],   minute=end_hm[1],   second=0, microsecond=0)
    return s, e

def asian_levels_by_day(df5: pd.DataFrame) -> pd.DataFrame:
    f5 = df5.copy(); f5.index = to_ch(f5.index)
    days = sorted({ts.date() for ts in f5.index})
    out = pd.DataFrame(index=f5.index, columns=["asian_high","asian_low"], dtype="float64")
    for d in days:
        s,e = _session_window(str(d),(1,0),(8,59))
        seg = f5.loc[(f5.index >= s) & (f5.index <= e)]
        if seg.empty: continue
        out.loc[f5.index.date == d, "asian_high"] = float(seg["high"].max())
        out.loc[f5.index.date == d, "asian_low"]  = float(seg["low"].min())
    return out

def london_levels_by_day(df5: pd.DataFrame) -> pd.DataFrame:
    f5 = df5.copy(); f5.index = to_ch(f5.index)
    days = sorted({ts.date() for ts in f5.index})
    out = pd.DataFrame(index=f5.index, columns=["london_high","london_low"], dtype="float64")
    for d in days:
        s,e = _session_window(str(d),(9,0),(15,29))
        seg = f5.loc[(f5.index >= s) & (f5.index <= e)]
        if seg.empty: continue
        out.loc[f5.index.date == d, "london_high"] = float(seg["high"].max())
        out.loc[f5.index.date == d, "london_low"]  = float(seg["low"].min())
    return out

def attach_session_levels(df5: pd.DataFrame,
                          asian_levels: pd.DataFrame,
                          london_levels: pd.DataFrame) -> pd.DataFrame:
    f5 = df5.copy(); f5.index = to_ch(f5.index)
    a = asian_levels.reindex(f5.index) if asian_levels is not None else None
    l = london_levels.reindex(f5.index) if london_levels is not None else None
    f5["asian_high"]  = a["asian_high"].astype("float64") if a is not None else np.nan
    f5["asian_low"]   = a["asian_low"].astype("float64")  if a is not None else np.nan
    f5["london_high"] = l["london_high"].astype("float64") if l is not None else np.nan
    f5["london_low"]  = l["london_low"].astype("float64")  if l is not None else np.nan
    return f5
