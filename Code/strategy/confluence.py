from __future__ import annotations
import pandas as pd
import numpy as np

def _ensure_numeric(df: pd.DataFrame, cols=("open","high","low","close")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ── Fair Value Gaps (5m) ────────────────────────────────────────────────────
def fvg_zones(df5: pd.DataFrame) -> pd.DataFrame:
    g = _ensure_numeric(df5, ("open","high","low","close"))
    z = []; idx = g.index
    for i in range(2, len(g)):
        c0 = g.iloc[i-2]; c2 = g.iloc[i]
        if float(c0["high"]) < float(c2["low"]):   # bullish
            z.append((idx[i], "bullish", float(c0["high"]), float(c2["low"])))
        if float(c0["low"])  > float(c2["high"]):  # bearish
            z.append((idx[i], "bearish", float(c2["high"]), float(c0["low"])))
    if not z:
        return pd.DataFrame(columns=["kind","z_low","z_high"]).astype({"kind":"object","z_low":"float64","z_high":"float64"})
    return pd.DataFrame(z, columns=["ts","kind","z_low","z_high"]).set_index("ts").sort_index()

# ── Order Blocks (5m, simple heuristic) ─────────────────────────────────────
def order_block_zones(df5: pd.DataFrame) -> pd.DataFrame:
    g = _ensure_numeric(df5); rows=[]; idx=g.index
    for i in range(1,len(g)):
        prev, curr = g.iloc[i-1], g.iloc[i]
        if prev["close"] < prev["open"] and curr["close"] > prev["high"]:
            z_low  = float(min(prev["open"], prev["close"]))
            z_high = float(prev["low"])
            rows.append((idx[i-1], "bullish", min(z_low,z_high), max(z_low,z_high)))
        if prev["close"] > prev["open"] and curr["close"] < prev["low"]:
            z_low  = float(prev["high"])
            z_high = float(max(prev["open"], prev["close"]))
            rows.append((idx[i-1], "bearish", min(z_low,z_high), max(z_low,z_high)))
    if not rows:
        return pd.DataFrame(columns=["kind","z_low","z_high"]).astype({"kind":"object","z_low":"float64","z_high":"float64"})
    return pd.DataFrame(rows, columns=["ts","kind","z_low","z_high"]).set_index("ts").sort_index()

# ── Breakers (5m, heuristic) ────────────────────────────────────────────────
def breaker_zones(df5: pd.DataFrame, lookahead: int = 3) -> pd.DataFrame:
    g = _ensure_numeric(df5); rows=[]; idx=g.index
    for i in range(1,len(g)):
        ref, cur = g.iloc[i-1], g.iloc[i]
        if cur["high"] > ref["high"]:
            later = g.iloc[i+1:min(i+1+lookahead, len(g))]
            if not later.empty and (later["close"] < ref["low"]).any():
                rows.append((idx[i-1], "bearish", float(max(ref["open"], ref["close"])), float(ref["high"])))
        if cur["low"] < ref["low"]:
            later = g.iloc[i+1:min(i+1+lookahead, len(g))]
            if not later.empty and (later["close"] > ref["high"]).any():
                rows.append((idx[i-1], "bullish", float(ref["low"]), float(min(ref["open"], ref["close"]))))
    if not rows:
        return pd.DataFrame(columns=["kind","z_low","z_high"]).astype({"kind":"object","z_low":"float64","z_high":"float64"})
    out = pd.DataFrame(rows, columns=["ts","kind","z_low","z_high"]).set_index("ts").sort_index()
    out[["z_low","z_high"]] = np.sort(out[["z_low","z_high"]], axis=1)
    return out

# ── Rolling EQ mid (informational) ──────────────────────────────────────────
def rolling_eq(df5: pd.DataFrame, window: int = 10) -> pd.Series:
    g = _ensure_numeric(df5, ("high","low"))
    mid = (g["high"].rolling(window).max() + g["low"].rolling(window).min()) / 2.0
    mid.name = "eq_mid"; return mid

# ── Zone touch helpers ──────────────────────────────────────────────────────
def _recent_zones(zdf: pd.DataFrame, current_ts, lookback_bars: int, all_index) -> pd.DataFrame:
    if zdf is None or zdf.empty or current_ts not in all_index:
        return zdf.iloc[0:0] if zdf is not None else zdf
    pos = all_index.get_loc(current_ts)
    past_index = all_index[max(0, pos - lookback_bars):pos]
    return zdf.loc[zdf.index.intersection(past_index)]

def _touch_any_zone(row_high, row_low, zones_df: pd.DataFrame) -> bool:
    if zones_df is None or zones_df.empty: return False
    z_low  = zones_df["z_low"].values; z_high = zones_df["z_high"].values
    return bool(np.any((row_low <= z_high) & (row_high >= z_low)))

def touched_bullish_retrace(df5, fvg5, eq5, ob5, brk5, lookback_bars: int = 48) -> pd.Series:
    g = _ensure_numeric(df5, ("high","low")); hits = pd.Series(False, index=g.index)
    for ts, row in g.iterrows():
        row_low, row_high = float(row["low"]), float(row["high"])
        any_hit = (
            _touch_any_zone(row_high, row_low, _recent_zones(ob5, ts, lookback_bars, g.index)) or
            _touch_any_zone(row_high, row_low, _recent_zones(brk5, ts, lookback_bars, g.index)) or
            _touch_any_zone(row_high, row_low, _recent_zones(fvg5, ts, lookback_bars, g.index))
        )
        hits.loc[ts] = bool(any_hit)
    return hits

def touched_bearish_retrace(df5, fvg5, eq5, ob5, brk5, lookback_bars: int = 48) -> pd.Series:
    return touched_bullish_retrace(df5, fvg5, eq5, ob5, brk5, lookback_bars)

# ── Simple 1m BOS (close breaks previous swing) ─────────────────────────────
def bos_1m_simple_close_break(m1: pd.DataFrame, start_ts, side: str):
    if start_ts is None: return None
    seq = m1.loc[m1.index > start_ts]
    if len(seq) < 2: return None
    prev_high = m1["high"].shift(1); prev_low = m1["low"].shift(1)
    for ts in seq.index:
        ph = prev_high.loc[ts] if ts in prev_high.index else np.nan
        pl = prev_low.loc[ts]  if ts in prev_low.index  else np.nan
        if not (np.isfinite(ph) and np.isfinite(pl)): continue
        if side == "low":
            if float(m1.loc[ts, "close"]) > float(ph): return ts
        else:
            if float(m1.loc[ts, "close"]) < float(pl): return ts
    return None

# ── Strict 50% EQ between sweep and retrace ─────────────────────────────────
def equilibrium_hit_50(m1_full: pd.DataFrame, sweep_ts, retrace_ts, want_long: bool) -> bool:
    if sweep_ts is None or retrace_ts is None: return False
    seg = m1_full.loc[(m1_full.index >= sweep_ts) & (m1_full.index <= retrace_ts)]
    if seg.empty or len(seg) < 2: return False
    if want_long:
        s = float(seg.iloc[0]["low"]); imp = float(seg["high"].max())
        if not (np.isfinite(s) and np.isfinite(imp) and imp > s): return False
        eq = s + 0.5*(imp - s)
        return float(seg.iloc[-1]["low"]) <= (eq + 1e-12)
    else:
        s = float(seg.iloc[0]["high"]); imp = float(seg["low"].min())
        if not (np.isfinite(s) and np.isfinite(imp) and s > imp): return False
        eq = s - 0.5*(s - imp)
        return float(seg.iloc[-1]["high"]) >= (eq - 1e-12)
