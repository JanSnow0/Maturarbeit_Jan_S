from __future__ import annotations
import numpy as np
import pandas as pd

from .common import ny_window_ch, to_ch

# ── 5m swing detector (simple 3-bar pivots) ─────────────────────────────────

def _swing_flags_5m(df5: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    swing high at i if high[i] >= high[i-1] and high[i] >= high[i+1]
    swing low  at i if low[i]  <= low[i-1]  and low[i]  <= low[i+1]
    """
    highs = df5["high"].astype(float).values
    lows  = df5["low"].astype(float).values
    n = len(df5)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            sh[i] = True
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            sl[i] = True
    return pd.Series(sh, index=df5.index), pd.Series(sl, index=df5.index)

# ── helpers for picking pre-NY and NY swings ────────────────────────────────

def _last_pre_ny_swing(df5: pd.DataFrame, day, kind: str):
    """Return (ts, price) of last swing HIGH/LOW before 15:30 CH on that day."""
    ny_s, _ = ny_window_ch(day)
    d = df5.loc[to_ch(df5.index) < ny_s]
    if d.empty:
        return None, np.nan
    sh, sl = _swing_flags_5m(d)
    if kind == "high":
        idx = d.index[sh]
        if len(idx) == 0:
            return None, np.nan
        ts = idx[-1]
        return ts, float(d.loc[ts, "high"])
    else:
        idx = d.index[sl]
        if len(idx) == 0:
            return None, np.nan
        ts = idx[-1]
        return ts, float(d.loc[ts, "low"])

def _first_ny_hour_swing(df5: pd.DataFrame, day, kind: str):
    """Return (ts, price) of FIRST swing HIGH/LOW inside 15:30–16:29 CH on that day."""
    ny_s, _ = ny_window_ch(day)
    ny_e_1h = ny_s + pd.Timedelta(hours=1) - pd.Timedelta(minutes=1)
    d = df5.loc[(to_ch(df5.index) >= ny_s) & (to_ch(df5.index) <= ny_e_1h)]
    if d.empty:
        return None, np.nan
    sh, sl = _swing_flags_5m(d)
    if kind == "high":
        idx = d.index[sh]
        if len(idx) == 0:
            return None, np.nan
        ts = idx[0]
        return ts, float(d.loc[ts, "high"])
    else:
        idx = d.index[sl]
        if len(idx) == 0:
            return None, np.nan
        ts = idx[0]
        return ts, float(d.loc[ts, "low"])

def _first_two_ny_swings(df5: pd.DataFrame, day, kind: str):
    """Scenario 3: take the first TWO swings (same kind) inside the first NY hour."""
    ny_s, _ = ny_window_ch(day)
    ny_e_1h = ny_s + pd.Timedelta(hours=1) - pd.Timedelta(minutes=1)
    d = df5.loc[(to_ch(df5.index) >= ny_s) & (to_ch(df5.index) <= ny_e_1h)]
    if d.empty:
        return (None, np.nan), (None, np.nan)
    sh, sl = _swing_flags_5m(d)
    if kind == "high":
        idx = d.index[sh]
        if len(idx) < 2:
            return (None, np.nan), (None, np.nan)
        ts1, ts2 = idx[0], idx[1]
        return (ts1, float(d.loc[ts1, "high"])), (ts2, float(d.loc[ts2, "high"]))
    else:
        idx = d.index[sl]
        if len(idx) < 2:
            return (None, np.nan), (None, np.nan)
        ts1, ts2 = idx[0], idx[1]
        return (ts1, float(d.loc[ts1, "low"])), (ts2, float(d.loc[ts2, "low"]))

# ── main SMT chooser ────────────────────────────────────────────────────────

def choose_symbol_for_day(
    df5_sym_refs: pd.DataFrame,
    df5_other_refs: pd.DataFrame,
    day,
    bias_dir: str,
    scenario: int,
) -> tuple[str, bool]:
    """
    Returns (chosen_symbol, require_strict_confirmation=False)

    SMT logic (5m):
      • Scenario 1/2: compare pre-NY swing (H1/L1) vs first NY-hour swing (H2/L2).
      • Scenario 3:   both swings come from the first NY hour (two consecutive swings).

    Bearish SMT: one HH (H2>H1) & the other LH (H2<H1) → short the LH index.
    Bullish SMT: one LL (L2<L1) & the other HL (L2>L1) → long the HL index.

    If no clean SMT, we return the 'sym' side (callers pass the symbol they intend to trade).
    """
    # We only need OHLC. Ensure the columns exist.
    for col in ["open", "high", "low", "close"]:
        if col not in df5_sym_refs.columns or col not in df5_other_refs.columns:
            raise ValueError(f"SMT chooser expects 5m OHLC columns; missing '{col}'")

    side = "bearish" if str(bias_dir).lower() == "bearish" else "bullish"

    if scenario in (1, 2):
        if side == "bearish":
            a1 = _last_pre_ny_swing(df5_sym_refs, day, "high")
            a2 = _first_ny_hour_swing(df5_sym_refs, day, "high")
            b1 = _last_pre_ny_swing(df5_other_refs, day, "high")
            b2 = _first_ny_hour_swing(df5_other_refs, day, "high")
            if all(np.isfinite([a1[1], a2[1], b1[1], b2[1]])):
                a_hh = a2[1] > a1[1]
                b_hh = b2[1] > b1[1]
                if a_hh and not b_hh:
                    return ("OTHER", False)  # short the LH index → OTHER
                if b_hh and not a_hh:
                    return ("SYM", False)    # short the LH index → SYM
        else:
            a1 = _last_pre_ny_swing(df5_sym_refs, day, "low")
            a2 = _first_ny_hour_swing(df5_sym_refs, day, "low")
            b1 = _last_pre_ny_swing(df5_other_refs, day, "low")
            b2 = _first_ny_hour_swing(df5_other_refs, day, "low")
            if all(np.isfinite([a1[1], a2[1], b1[1], b2[1]])):
                a_ll = a2[1] < a1[1]
                b_ll = b2[1] < b1[1]
                if a_ll and not b_ll:
                    return ("OTHER", False)  # long the HL index → OTHER
                if b_ll and not a_ll:
                    return ("SYM", False)    # long the HL index → SYM
    else:
        # scenario 3: both swings inside NY hour
        if side == "bearish":
            a1, a2 = _first_two_ny_swings(df5_sym_refs, day, "high")
            b1, b2 = _first_two_ny_swings(df5_other_refs, day, "high")
            if all(np.isfinite([a1[1], a2[1], b1[1], b2[1]])):
                a_hh = a2[1] > a1[1]
                b_hh = b2[1] > b1[1]
                if a_hh and not b_hh:
                    return ("OTHER", False)
                if b_hh and not a_hh:
                    return ("SYM", False)
        else:
            a1, a2 = _first_two_ny_swings(df5_sym_refs, day, "low")
            b1, b2 = _first_two_ny_swings(df5_other_refs, day, "low")
            if all(np.isfinite([a1[1], a2[1], b1[1], b2[1]])):
                a_ll = a2[1] < a1[1]
                b_ll = b2[1] < b1[1]
                if a_ll and not b_ll:
                    return ("OTHER", False)
                if b_ll and not a_ll:
                    return ("SYM", False)

    # No clean SMT → keep caller’s symbol ("SYM")
    return ("SYM", False)
