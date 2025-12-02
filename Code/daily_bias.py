from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Settings: Swiss time (Europe/Zurich) and session windows
# ─────────────────────────────────────────────────────────────────────────────
CH_TZ = "Europe/Zurich"
ASIAN_START = 1.0    # 01:00
ASIAN_END   = 9.0    # 08:59
LONDON_START= 9.0    # 09:00
LONDON_END  = 15.5   # 15:29
NY_START    = 15.5   # 15:30
NY_END      = 22.0   # 22:00

# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_ch(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx_utc.tz_convert(CH_TZ)

def _swiss_day(idx_utc: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(_to_ch(idx_utc).date, index=idx_utc, name="day")

def _session_label(idx_utc: pd.DatetimeIndex) -> pd.Series:
    idx_ch = _to_ch(idx_utc)
    h = idx_ch.hour + idx_ch.minute/60.0
    asian  = (h >= ASIAN_START)  & (h < ASIAN_END)
    london = (h >= LONDON_START) & (h < LONDON_END)
    ny     = (h >= NY_START)     & (h < NY_END)
    out = np.where(asian, "ASIAN",
          np.where(london, "LONDON",
          np.where(ny, "NY", "OFF")))
    return pd.Series(out, index=idx_utc, name="session")

def _ny_window_ch(day):
    start = pd.Timestamp(day, tz=CH_TZ).replace(hour=15, minute=30, second=0, microsecond=0)
    end   = pd.Timestamp(day, tz=CH_TZ).replace(hour=22, minute=0, second=0, microsecond=0)
    return start, end

# ─────────────────────────────────────────────────────────────────────────────
# Build finished 5m / 1h bars from live bundle columns
# (expects *_5m_live and *_1h_live columns)
# ─────────────────────────────────────────────────────────────────────────────
def _finished_5m(df_live: pd.DataFrame) -> pd.DataFrame:
    need = {"open_5m_live","high_5m_live","low_5m_live","close_5m_live"}
    miss = need - set(df_live.columns)
    if miss:
        raise ValueError(f"Missing 5m columns: {miss}")
    out = df_live.loc[df_live["close_5m_live"].notna(), list(need)].copy()
    out.columns = ["open","high","low","close"]
    return out

def _finished_1h(df_live: pd.DataFrame) -> pd.DataFrame:
    need = {"open_1h_live","high_1h_live","low_1h_live","close_1h_live"}
    miss = need - set(df_live.columns)
    if miss:
        raise ValueError(f"Missing 1h columns: {miss}")
    out = df_live.loc[df_live["close_1h_live"].notna(), list(need)].copy()
    out.columns = ["open","high","low","close"]
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Reference levels: previous hour, Asian session of same Swiss day
# ─────────────────────────────────────────────────────────────────────────────
def _attach_prev_hour_levels(df5: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    h_prev = df1h[["high","low"]].shift(1).dropna().rename(
        columns={"high":"prev_hour_high","low":"prev_hour_low"}
    )
    a = df5.reset_index().rename(columns={"timestamp":"ts"}).sort_values("ts")
    b = h_prev.reset_index().rename(columns={"timestamp":"ts"}).sort_values("ts")
    out = pd.merge_asof(a, b, on="ts", direction="backward")
    out = out.set_index("ts"); out.index.name = "timestamp"
    return out

def _levels_by_session(df5_all: pd.DataFrame, which: str) -> pd.DataFrame:
    ses = _session_label(df5_all.index); day = _swiss_day(df5_all.index)
    d = pd.DataFrame({
        "day": day.values, "session": ses.values,
        "high": df5_all["high"].values, "low": df5_all["low"].values
    }, index=df5_all.index)
    g = d[d["session"]==which].groupby("day").agg(
        **{f"{which.lower()}_high":("high","max"), f"{which.lower()}_low":("low","min")}
    )
    return g

def _asian_levels_by_day(df5_all: pd.DataFrame) -> pd.DataFrame:
    return _levels_by_session(df5_all, "ASIAN")

def _attach_asian_levels(df5: pd.DataFrame, asian_lvls: pd.DataFrame) -> pd.DataFrame:
    day = _swiss_day(df5.index)
    out = df5.copy()
    out["day"] = day.values
    out = out.join(asian_lvls, on="day")
    return out

def _is_weekday(d) -> bool:
    return pd.Timestamp(d).dayofweek < 5

# ─────────────────────────────────────────────────────────────────────────────
# Sweep detection during a window
# ─────────────────────────────────────────────────────────────────────────────
def _find_sweeps(df5_refs: pd.DataFrame, start_ch: pd.Timestamp, end_ch: pd.Timestamp):
    g = df5_refs.loc[
        (_to_ch(df5_refs.index) >= start_ch) & (_to_ch(df5_refs.index) <= end_ch)
    ].copy()
    if g.empty:
        return pd.DataFrame(columns=["ts","side"])
    ref_hi = pd.concat([g["prev_hour_high"], g["asian_high"]], axis=1).max(axis=1)
    ref_lo = pd.concat([g["prev_hour_low"],  g["asian_low"]],  axis=1).min(axis=1)
    took_high = g["high"] > ref_hi
    took_low  = g["low"]  < ref_lo
    hits = g.loc[took_high | took_low]
    if hits.empty:
        return pd.DataFrame(columns=["ts","side"])
    sides = np.where(took_high.loc[hits.index], "high", "low")
    out = pd.DataFrame({"ts": hits.index, "side": sides})
    return out

# ─────────────────────────────────────────────────────────────────────────────
# NEW: BOS after sweep using running extremes (no pivot needed)
# ─────────────────────────────────────────────────────────────────────────────
def _bos_5m_after_sweep_running_extreme(df5: pd.DataFrame, sweep_ts, side: str,
                                        strict_color: bool = True,
                                        london_end_ch: pd.Timestamp | None = None):
    """
    BOS nach Sweep mit laufendem Extrem (keine Pivot-Bestätigung nötig):
      • High-Sweep (bärisch): tracke laufendes Tief seit Sweep; BOS wenn Close < running_low
        (BOS-Kerze rot, wenn strict_color=True).
      • Low-Sweep  (bullisch): tracke laufendes Hoch seit Sweep; BOS wenn Close > running_high
        (BOS-Kerze grün, wenn strict_color=True).
    """
    if sweep_ts not in df5.index:
        return None

    # Bars ab dem Sweep (einschließlich) holen
    after = df5.loc[df5.index >= sweep_ts].copy()
    if after.empty:
        return None

    # Optional bis London-Ende beschneiden
    if london_end_ch is not None:
        idx_ch = after.index.tz_convert(CH_TZ)
        after = after.loc[idx_ch <= london_end_ch]
        if after.empty:
            return None

    running_low  = None
    running_high = None

    for ts, row in after.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

        if side == "high":
            # bärischer BOS: aktualisiere laufendes Tief
            running_low = l if running_low is None else min(running_low, l)
            # BOS-Check: Close unter running_low (Kerze idealerweise rot)
            color_ok = (c < o) if strict_color else True
            if running_low is not None and c < running_low and color_ok:
                return ts

        elif side == "low":
            # bullischer BOS: aktualisiere laufendes Hoch
            running_high = h if running_high is None else max(running_high, h)
            # BOS-Check: Close über running_high (Kerze idealerweise grün)
            color_ok = (c > o) if strict_color else True
            if running_high is not None and c > running_high and color_ok:
                return ts

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Daily bias computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_daily_bias(in_csv: Path) -> pd.DataFrame:
    # Load live bundle (Swiss-time CSV with timestamp as first column, UTC tz)
    df = pd.read_csv(in_csv)
    # parse timestamp
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    ts = pd.to_datetime(df.pop(ts_col), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Unparsable timestamps in input.")
    df.index = pd.DatetimeIndex(ts, name="timestamp")

    # coerce numeric
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build 5m/1h finished bars and attach reference levels
    df5  = _finished_5m(df)
    df1h = _finished_1h(df)
    refs = _attach_prev_hour_levels(df5, df1h)
    asian = _asian_levels_by_day(df5)
    refs = _attach_asian_levels(refs, asian)

    # Iterate weekdays
    days = sorted({d for d in _swiss_day(df.index).unique() if _is_weekday(d)})
    rows = []
    for d in days:
        # London window
        ld_s = pd.Timestamp(d, tz=CH_TZ).replace(hour=9,  minute=0, second=0, microsecond=0)
        ld_e = pd.Timestamp(d, tz=CH_TZ).replace(hour=15, minute=29, second=0, microsecond=0)

        # Find London sweeps (vs. prev-hour and Asian levels)
        sweeps_ldn = _find_sweeps(refs, ld_s, ld_e)

        if not sweeps_ldn.empty:
            # take the LAST London sweep
            last_side = sweeps_ldn.iloc[-1]["side"]   # "high" or "low"
            last_ts   = sweeps_ldn.iloc[-1]["ts"]

            # bias opposite to the last sweep
            bias = "bullish" if last_side == "low" else "bearish"

            # 5m BOS after sweep using running extremes (within London)
            df5_day = df5.loc[
                (_to_ch(df5.index) >= ld_s) & (_to_ch(df5.index) <= ld_e),
                ["open","high","low","close"]
            ]
            bos_ts = _bos_5m_after_sweep_running_extreme(
                df5_day, last_ts, side=last_side,
                strict_color=True,        # set to False for a looser BOS
                london_end_ch=ld_e
            )
            scenario = 2 if bos_ts is not None else 1

        else:
            # No London sweep → Scenario 3 via NY
            ny_s, ny_e = _ny_window_ch(d)
            sweeps_ny = _find_sweeps(refs, ny_s, ny_e)
            if sweeps_ny.empty:
                # No sweep at all → skip day (no bias)
                continue
            first_side = sweeps_ny.iloc[0]["side"]
            bias = "bullish" if first_side == "low" else "bearish"
            scenario = 3

        rows.append({"day": d, "scenario": scenario, "bias": bias})

    out = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Swiss-time live bundle CSV (with 1m_5m_1h_live columns)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV with day,scenario,bias")
    args = ap.parse_args()

    out = compute_daily_bias(Path(args.inp))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out} with {len(out)} rows")

if __name__ == "__main__":
    main()
