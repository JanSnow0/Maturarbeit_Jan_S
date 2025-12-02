from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import random

from .common import ny_window_ch, to_ch, finished_5m, finished_1h, attach_prev_hour_levels, asian_levels_by_day, london_levels_by_day, attach_session_levels
from .confluence import fvg_zones, rolling_eq, order_block_zones, breaker_zones, touched_bullish_retrace, touched_bearish_retrace, bos_1m_simple_close_break, equilibrium_hit_50
from .smt import choose_symbol_for_day

@dataclass
class RiskConfig:
    start_equity: float = 10000.0
    buffer: float = 0.0

# NY **entry** timebox now 15:30–22:00 CH
TIMEBOX_START_HM = (15, 30)
TIMEBOX_END_HM   = (22, 0)

def _extract_1m(df_live_sym: pd.DataFrame) -> pd.DataFrame:
    cands = [
        {"o":"open","h":"high","l":"low","c":"close"},
        {"o":"open_1m","h":"high_1m","l":"low_1m","c":"close_1m"},
        {"o":"open_1m_live","h":"high_1m_live","l":"low_1m_live","c":"close_1m_live"},
        {"o":"o","h":"h","l":"l","c":"c"},
    ]
    for m in cands:
        if all(v in df_live_sym.columns for v in m.values()):
            m1 = df_live_sym[[m["o"],m["h"],m["l"],m["c"]]].copy()
            m1.columns = ["open","high","low","close"]
            for c in ["open","high","low","close"]:
                m1[c] = pd.to_numeric(m1[c], errors="coerce")
            return m1
    raise ValueError("Could not find 1m OHLC columns.")

def _day_slice(df: pd.DataFrame, day: str, pad_hours: int = 2) -> pd.DataFrame:
    idx_ch = to_ch(df.index)
    d0 = pd.Timestamp(day, tz="Europe/Zurich").replace(hour=0, minute=0, second=0, microsecond=0)
    d1 = d0 + pd.Timedelta(days=1)
    s = d0 - pd.Timedelta(hours=pad_hours)
    e = d1 + pd.Timedelta(hours=pad_hours)
    return df.loc[(idx_ch >= s) & (idx_ch <= e)]

def _prep_refs(df_live_sym: pd.DataFrame, day: str):
    sub = _day_slice(df_live_sym, day, pad_hours=2)
    if sub.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    m1 = _extract_1m(sub)
    ny_s, ny_e = ny_window_ch(day)
    # We still compute standard NY window; entry check uses TIMEBOX_* below
    m1_ny = m1.loc[(to_ch(m1.index) >= ny_s) & (to_ch(m1.index) <= ny_e)]

    df5 = finished_5m(sub)
    df1h = finished_1h(sub)
    df5a = attach_prev_hour_levels(df5, df1h)
    asian = asian_levels_by_day(df5)
    london = london_levels_by_day(df5)
    df5a = attach_session_levels(df5a, asian, london)
    f5_ny = df5a.loc[(to_ch(df5a.index) >= ny_s) & (to_ch(df5a.index) <= ny_e)]

    return m1, m1_ny, f5_ny, df5a

def _in_timebox(ts, day):
    if ts is None: return False
    ch = ts.tz_convert("Europe/Zurich")
    start = ch.replace(hour=TIMEBOX_START_HM[0], minute=TIMEBOX_START_HM[1], second=0, microsecond=0)
    end   = ch.replace(hour=TIMEBOX_END_HM[0],   minute=TIMEBOX_END_HM[1],   second=0, microsecond=0)
    return (ch >= start) and (ch <= end)

def _sweep_london(df5_refs: pd.DataFrame, day):
    CH = "Europe/Zurich"
    ld_s = pd.Timestamp(day, tz=CH).replace(hour=9, minute=0)
    ld_e = pd.Timestamp(day, tz=CH).replace(hour=15, minute=29)
    g = df5_refs.copy(); idx_ch = g.index.tz_convert(CH)
    ref_hi = pd.concat([g["prev_hour_high"], g["asian_high"]], axis=1).max(axis=1)
    ref_lo = pd.concat([g["prev_hour_low"],  g["asian_low"]],  axis=1).min(axis=1)
    gg = g.loc[(idx_ch >= ld_s) & (idx_ch <= ld_e)]
    if gg.empty: return (None, None)
    took_high = gg["high"] > ref_hi.loc[gg.index]
    took_low  = gg["low"]  < ref_lo.loc[gg.index]
    hits = gg.loc[took_high | took_low]
    if hits.empty: return (None, None)
    side = "high" if bool(took_high.loc[hits.index[-1]]) else "low"
    return (hits.index[-1], side)

def _sweep_ny(df5_refs: pd.DataFrame, day):
    CH = "Europe/Zurich"
    ny_s = pd.Timestamp(day, tz=CH).replace(hour=15, minute=30)
    ny_e = pd.Timestamp(day, tz=CH).replace(hour=TIMEBOX_END_HM[0], minute=TIMEBOX_END_HM[1])
    g = df5_refs.copy(); idx_ch = g.index.tz_convert(CH)
    ref_hi = pd.concat([g["prev_hour_high"], g["asian_high"]], axis=1).max(axis=1)
    ref_lo = pd.concat([g["prev_hour_low"],  g["asian_low"]],  axis=1).min(axis=1)
    gg = g.loc[(idx_ch >= ny_s) & (idx_ch <= ny_e)]
    if gg.empty: return (None, None)
    took_high = gg["high"] > ref_hi.loc[gg.index]
    took_low  = gg["low"]  < ref_lo.loc[gg.index]
    hits = gg.loc[took_high | took_low]
    if hits.empty: return (None, None)
    side = "high" if bool(took_high.loc[hits.index[0]]) else "low"
    return (hits.index[0], side)

def _sl_on_retrace(retrace_row: pd.Series, want_long: bool):
    if retrace_row is None or retrace_row.empty: return np.nan
    low = float(retrace_row["low"]); high = float(retrace_row["high"])
    return low if want_long else high

def _tp1_impulse_extreme(m1_full: pd.DataFrame, sweep_ts, retrace_ts, want_long: bool):
    if sweep_ts is None or retrace_ts is None: return np.nan
    seg = m1_full.loc[(m1_full.index >= sweep_ts) & (m1_full.index < retrace_ts)]
    if seg.empty:
        seg = m1_full.loc[(m1_full.index >= sweep_ts) & (m1_full.index <= retrace_ts)]
    if seg.empty: return np.nan
    return float(seg["high"].max() if want_long else seg["low"].min())

def _tp_chain(entry_px: float, stop: float, tp1_cand: float, f5_before: pd.DataFrame, want_long: bool):
    """
    TP rules (strict):
      - TP1 = impulse extreme (fallback to 1R if not beyond entry)
      - TP2 exists ONLY if it is further than TP1
      - TP3 exists ONLY if TP2 exists AND is further than TP2
      - If only TP1 => 100% at TP1
      - If TP1 & TP2 => 50% / 50%
      - If TP1, TP2, TP3 => 50% / 25% / 25%
    """
    if not (np.isfinite(entry_px) and np.isfinite(stop)):
        return (np.nan, np.nan, np.nan)

    if want_long:
        # TP1
        tp1 = tp1_cand if (np.isfinite(tp1_cand) and tp1_cand > entry_px) else (entry_px + abs(entry_px - stop))
        # TP2 candidate: London High
        ldnH = f5_before["london_high"].dropna().max() if "london_high" in f5_before else np.nan
        tp2 = float(ldnH) if (np.isfinite(ldnH) and ldnH > tp1) else np.nan
        # TP3 candidate: Asian High BUT ONLY if tp2 exists and tp3 > tp2
        asH  = f5_before["asian_high"].dropna().max()  if "asian_high"  in f5_before else np.nan
        tp3 = float(asH) if (np.isfinite(tp2) and np.isfinite(asH) and asH > tp2) else np.nan
    else:
        tp1 = tp1_cand if (np.isfinite(tp1_cand) and tp1_cand < entry_px) else (entry_px - abs(entry_px - stop))
        ldnL = f5_before["london_low"].dropna().min() if "london_low" in f5_before else np.nan
        tp2 = float(ldnL) if (np.isfinite(ldnL) and ldnL < tp1) else np.nan
        asL  = f5_before["asian_low"].dropna().min()  if "asian_low"  in f5_before else np.nan
        tp3 = float(asL) if (np.isfinite(tp2) and np.isfinite(asL) and asL < tp2) else np.nan

    return (float(tp1), (float(tp2) if np.isfinite(tp2) else np.nan),
                    (float(tp3) if np.isfinite(tp3) else np.nan))

def _choose_symbol_by_smt_or_retrace(df5_spx_refs: pd.DataFrame, df5_nsx_refs: pd.DataFrame,
                                     f5_spx_ny: pd.DataFrame, f5_nsx_ny: pd.DataFrame,
                                     retr_spx: pd.Series, retr_nsx: pd.Series,
                                     day: str, bias_dir: str):
    # Try SMT first
    sym, strict = choose_symbol_for_day(df5_spx_refs, df5_nsx_refs, day, bias_dir, scenario=0)
    if sym in ("SPX","NSX"):
        return sym
    # No SMT → hard 50/50 coin flip (as requested)
    return random.choice(["SPX","NSX"])

class ScenarioBase:
    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def _run_common(self, symbol, other_symbol,
                    df_live_sym, df_live_other,
                    df5_sym_refs, df5_other_refs,
                    bias_row: pd.Series, *, scenario_id: int):

        day = str(bias_row["day"])
        bias = str(bias_row.get("bias","")).lower()
        if bias not in ("bullish","bearish"): return []
        want_long = (bias == "bullish")

        m1_full, m1_ny, f5_ny, df5_refs = _prep_refs(df_live_sym, day)
        if m1_full.empty or f5_ny.empty:
            return []

        fvg5 = fvg_zones(f5_ny); eq5 = rolling_eq(f5_ny, 10)
        ob5 = order_block_zones(f5_ny); brk5 = breaker_zones(f5_ny)
        retr_flags = touched_bullish_retrace(f5_ny, fvg5, eq5, ob5, brk5) if want_long else \
                     touched_bearish_retrace(f5_ny, fvg5, eq5, ob5, brk5)

        m1_full_other, m1_ny_other, f5_ny_other, df5_refs_other = _prep_refs(df_live_other, day)
        retr_flags_other = touched_bullish_retrace(f5_ny_other, fvg5, eq5, ob5, brk5) if want_long else \
                           touched_bearish_retrace(f5_ny_other, fvg5, eq5, ob5, brk5)

        chosen = _choose_symbol_by_smt_or_retrace(
            df5_sym_refs, df5_other_refs, f5_ny, f5_ny_other,
            retr_flags, retr_flags_other, day, bias
        )
        if chosen == "NSX":
            symbol, other_symbol = other_symbol, symbol
            df_live_sym, df_live_other = df_live_other, df_live_sym
            df5_sym_refs, df5_other_refs = df5_other_refs, df5_sym_refs
            m1_full, m1_ny, f5_ny, df5_refs = m1_full_other, m1_ny_other, f5_ny_other, df5_refs_other
            retr_flags = retr_flags_other

        if scenario_id in (1,2):
            sweep_ts, sweep_side = _sweep_london(df5_refs, day)
        else:
            sweep_ts, sweep_side = _sweep_ny(df5_refs, day)
        if sweep_ts is None:
            return []

        cand_ts_list = retr_flags[retr_flags].index.tolist()
        if not cand_ts_list:
            return []
        retr_ts = cand_ts_list[0]
        retr_row = f5_ny.loc[retr_ts]

        if not equilibrium_hit_50(m1_full, sweep_ts, retr_ts, want_long):
            return []

        dir_side = "low" if want_long else "high"
        bos_ts = bos_1m_simple_close_break(m1_full, retr_ts, dir_side)
        # Entry must be inside 15:30–22:00 CH
        if bos_ts is None or (not _in_timebox(bos_ts, day)):
            return []

        if bos_ts not in m1_full.index:
            return []
        entry_px = float(m1_full.loc[bos_ts, "close"])
        stop = _sl_on_retrace(retr_row, want_long)

        f5_before = f5_ny.loc[f5_ny.index <= bos_ts]
        tp1 = _tp1_impulse_extreme(m1_full, sweep_ts, retr_ts, want_long)
        tp1, tp2, tp3 = _tp_chain(entry_px, stop, tp1, f5_before, want_long)

        return [{
            "symbol": symbol, "day": day, "scenario": scenario_id, "bias": bias,
            "recipe": f"S{scenario_id}_NY: 5m retrace + 1m BOS",
            "entry_time_utc": bos_ts.tz_convert("UTC"), "entry_price": float(entry_px),
            "stop": float(stop), "tp1": float(tp1),
            "tp2": (float(tp2) if np.isfinite(tp2) else np.nan),
            "tp3": (float(tp3) if np.isfinite(tp3) else np.nan),
        }]

class Scenario1NY(ScenarioBase):
    def run_day(self, symbol, other_symbol,
                df_live_sym, df_live_other,
                df5_sym_refs, df5_other_refs,
                bias_row: pd.Series):
        return self._run_common(symbol, other_symbol,
                                df_live_sym, df_live_other,
                                df5_sym_refs, df5_other_refs,
                                bias_row, scenario_id=1)

class Scenario2NY(ScenarioBase):
    def run_day(self, symbol, other_symbol,
                df_live_sym, df_live_other,
                df5_sym_refs, df5_other_refs,
                bias_row: pd.Series):
        return self._run_common(symbol, other_symbol,
                                df_live_sym, df_live_other,
                                df5_sym_refs, df5_other_refs,
                                bias_row, scenario_id=2)

class Scenario3NY(ScenarioBase):
    def run_day(self, symbol, other_symbol,
                df_live_sym, df_live_other,
                df5_sym_refs, df5_other_refs,
                bias_row: pd.Series):
        return self._run_common(symbol, other_symbol,
                                df_live_sym, df_live_other,
                                df5_sym_refs, df5_other_refs,
                                bias_row, scenario_id=3)
