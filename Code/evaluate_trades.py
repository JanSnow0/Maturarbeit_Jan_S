# Code/evaluate_trades.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CH_TZ = "Europe/Zurich"

# ------------------- Loader 1m (für Level-Treffer) -------------------

def load_1m_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_candidates = [c for c in df.columns if c.lower() in ("ts","time","timestamp","datetime","date")]
    if not ts_candidates:
        raise ValueError(f"Kein Zeitstempel in {path}")
    ts = pd.to_datetime(df[ts_candidates[0]], utc=True, errors="coerce").dt.tz_convert(CH_TZ)
    df = df.set_index(ts).sort_index()
    cands = [
        ("open","high","low","close"),
        ("open_1m","high_1m","low_1m","close_1m"),
        ("o","h","l","c"),
        ("Open","High","Low","Close"),
    ]
    found = None
    for o,h,l,c in cands:
        if all(x in df.columns for x in (o,h,l,c)):
            found = (o,h,l,c); break
    if found is None:
        raise ValueError(f"Keine 1m-OHLC Spalten in {path}")
    o,h,l,c = found
    out = df[[o,h,l,c]].copy()
    out.columns = ["open","high","low","close"]
    for col in ["open","high","low","close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

# ------------------- Helpers -------------------

def to_ch(ts: pd.Timestamp) -> pd.Timestamp:
    if ts is None or pd.isna(ts): return ts
    if ts.tzinfo is None: return ts.tz_localize("UTC").tz_convert(CH_TZ)
    return ts.tz_convert(CH_TZ)

def trade_day_window_ch(entry_ts_utc: pd.Timestamp) -> tuple[pd.Timestamp,pd.Timestamp]:
    ch = to_ch(entry_ts_utc)
    start = ch
    end   = ch.replace(hour=22, minute=0, second=0, microsecond=0)
    if end < start:
        end = end + pd.Timedelta(days=1)
    return start, end

def first_time_hit_upside(bars: pd.DataFrame, level: float) -> pd.Timestamp | None:
    if not np.isfinite(level): return None
    hit = bars.index[bars["high"] >= level]
    return hit[0] if len(hit) else None

def first_time_hit_downside(bars: pd.DataFrame, level: float) -> pd.Timestamp | None:
    if not np.isfinite(level): return None
    hit = bars.index[bars["low"] <= level]
    return hit[0] if len(hit) else None

def compute_partials_R(entry: float, stop: float, tp1: float, tp2: float, tp3: float,
                       want_long: bool, bars_after_entry: pd.DataFrame) -> tuple[str, float]:
    """TP1 50% → SL=BE → TP2 25% → TP3 25% (Pfadprüfung auf 1m bis 22:00 CH)."""
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        return ("invalid", np.nan)

    if want_long:
        t_stop = first_time_hit_downside(bars_after_entry, stop)
        t_tp1  = first_time_hit_upside(bars_after_entry, tp1)
        t_tp2  = first_time_hit_upside(bars_after_entry, tp2)
        t_tp3  = first_time_hit_upside(bars_after_entry, tp3)
    else:
        t_stop = first_time_hit_upside(bars_after_entry, stop)
        t_tp1  = first_time_hit_downside(bars_after_entry, tp1)
        t_tp2  = first_time_hit_downside(bars_after_entry, tp2)
        t_tp3  = first_time_hit_downside(bars_after_entry, tp3)

    candidates = []
    if t_stop is not None: candidates.append(("stop", t_stop))
    if t_tp1  is not None: candidates.append(("tp1",  t_tp1))
    if t_tp2  is not None: candidates.append(("tp2",  t_tp2))
    if t_tp3  is not None: candidates.append(("tp3",  t_tp3))
    first_hit = sorted(candidates, key=lambda x: x[1])[0][0] if candidates else "none"

    # Stop vor TP1 → -1R
    if (t_stop is not None) and (t_tp1 is None or t_stop < t_tp1):
        return (first_hit, -1.0)

    # TP1 vor Stop → +0.5R, dann SL=BE
    if (t_tp1 is not None) and (t_stop is None or t_tp1 <= t_stop):
        total_R = 0.5 * (abs(tp1 - entry) / risk)
        after_tp1 = bars_after_entry.loc[bars_after_entry.index >= t_tp1]
        if after_tp1.empty: return (first_hit, float(total_R))

        if want_long:
            t_be = first_time_hit_downside(after_tp1, entry)
            if t_tp2 is not None and t_tp2 >= t_tp1 and (t_be is None or t_tp2 <= t_be):
                total_R += 0.25 * (abs(tp2 - entry) / risk)
                after_tp2 = after_tp1.loc[after_tp1.index >= t_tp2]
                t_be2 = first_time_hit_downside(after_tp2, entry)
                if t_tp3 is not None and t_tp3 >= t_tp2 and (t_be2 is None or t_tp3 <= t_be2):
                    total_R += 0.25 * (abs(tp3 - entry) / risk)
        else:
            t_be = first_time_hit_upside(after_tp1, entry)
            if t_tp2 is not None and t_tp2 >= t_tp1 and (t_be is None or t_tp2 <= t_be):
                total_R += 0.25 * (abs(tp2 - entry) / risk)
                after_tp2 = after_tp1.loc[after_tp1.index >= t_tp2]
                t_be2 = first_time_hit_upside(after_tp2, entry)
                if t_tp3 is not None and t_tp3 >= t_tp2 and (t_be2 is None or t_tp3 <= t_be2):
                    total_R += 0.25 * (abs(tp3 - entry) / risk)
        return (first_hit, float(total_R))

    # Weder Stop noch TP1 → 0R
    return (first_hit, 0.0)

def filter_wrong_side(trades: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Entfernt Trades mit Stop auf falscher Seite (Long: SL<Entry, Short: SL>Entry)."""
    df = trades.copy()
    if "bias" not in df.columns: return (df, 0)
    df["bias_norm"] = df["bias"].astype(str).str.lower().str.strip()
    before = len(df)
    good_long  = (df["bias_norm"] == "bullish") & (df["stop"] <  df["entry_price"])
    good_short = (df["bias_norm"] == "bearish") & (df["stop"] >  df["entry_price"])
    df = df.loc[good_long | good_short].drop(columns=["bias_norm"])
    return (df, before - len(df))

def to_daily_equity_from_trades(trades: pd.DataFrame, start_cash: float, risk_per_trade: float) -> pd.Series:
    """
    Baut eine tägliche Equity-Kurve aus chronologischen Trades.
    Equity_{i+1} = Equity_i + R_multiple_i * risk_per_trade
    Danach auf Tage abbilden und vorwärts füllen.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    t = trades.copy()
    t["entry_time_utc"] = pd.to_datetime(t["entry_time_utc"], utc=True, errors="coerce")
    t = t.sort_values("entry_time_utc")
    eq_vals = []
    eq = start_cash
    for _, row in t.iterrows():
        r = float(row.get("R_multiple", 0.0))
        if not np.isfinite(r): r = 0.0
        eq = eq + r * risk_per_trade
        eq_vals.append(eq)
    t["equity_after"] = eq_vals

    t["entry_time_ch"] = t["entry_time_utc"].dt.tz_convert(CH_TZ)
    t["day"] = t["entry_time_ch"].dt.normalize()

    daily = t.groupby("day")["equity_after"].last().sort_index()

    # Lückenfüllung für stabile Vol/DD-Berechnung
    full_days = pd.date_range(daily.index[0], daily.index[-1], freq="1D", tz=CH_TZ)
    equity = pd.Series(index=full_days, dtype=float)
    equity.loc[daily.index] = daily.values
    equity.iloc[0] = start_cash if np.isnan(equity.iloc[0]) else equity.iloc[0]
    equity = equity.ffill()
    equity.name = "equity"
    return equity

def metrics_simple(equity: pd.Series) -> dict:
    """
    Liefert:
      - total_return_amount, total_return_pct
      - max_drawdown_amount, max_drawdown_pct
      - daily_vol (Stdabw. täglicher Renditen, nicht annualisiert)
      - return_risk_ratio = total_return_amount / |max_drawdown_amount|
    """
    equity = equity.dropna()
    if equity.empty:
        return {k: np.nan for k in [
            "start_value","end_value","total_return_amount","total_return_pct",
            "max_drawdown_amount","max_drawdown_pct","daily_vol","return_risk_ratio","n_days"
        ]}
    start_v = float(equity.iloc[0])
    end_v   = float(equity.iloc[-1])
    total_return_amount = end_v - start_v
    total_return_pct    = (end_v / start_v - 1.0) if start_v > 0 else np.nan

    # tägliche Renditen
    rets = equity.pct_change().dropna()
    daily_vol = float(rets.std(ddof=1)) if len(rets) > 1 else np.nan

    # Max Drawdown (Betrag & %)
    roll_max = equity.cummax()
    dd_amt_series = roll_max - equity
    max_dd_amount = float(dd_amt_series.max()) if len(dd_amt_series) else 0.0

    dd_pct_series = equity / roll_max - 1.0
    max_dd_pct    = float(dd_pct_series.min()) if len(dd_pct_series) else 0.0  # negativ

    # Ertrag/Risiko (Betrag/Betrag)
    rr = (total_return_amount / max_dd_amount) if (max_dd_amount and max_dd_amount > 0) else np.nan

    return {
        "start_value": start_v,
        "end_value": end_v,
        "total_return_amount": total_return_amount,
        "total_return_pct": total_return_pct,
        "max_drawdown_amount": max_dd_amount,
        "max_drawdown_pct": max_dd_pct,
        "daily_vol": daily_vol,
        "return_risk_ratio": rr,
        "n_days": int(len(equity)),
    }

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="Trades-CSV (entry_time_utc, entry_price, stop, tp1..tp3, bias, symbol)")
    ap.add_argument("--spx", required=True, help="SPX 1m CSV (CH Zeit), für Level-Hits (wie gehabt)")
    ap.add_argument("--nsx", required=True, help="NSX 1m CSV (CH Zeit)")
    ap.add_argument("--out", required=True, help="Output Trades+Ergebnis CSV")
    ap.add_argument("--out_equity", required=True, help="Output tägliche Equity-Kurve CSV")
    ap.add_argument("--start_cash", type=float, default=10000.0, help="Startkapital für Equity-Kurve (Default 10000)")
    ap.add_argument("--risk_per_trade", type=float, default=1000.0, help="Risikobetrag je Trade (für R_multiple → $)")
    args = ap.parse_args()

    trades_path = Path(args.trades)
    spx_path    = Path(args.spx)
    nsx_path    = Path(args.nsx)
    out_path    = Path(args.out)
    out_eq_path = Path(args.out_equity)

    # Trades laden
    trades = pd.read_csv(trades_path)

    # Filter: Stop auf falscher Seite raus
    trades, removed = filter_wrong_side(trades)

    if trades.empty:
        trades.assign(first_hit="none", R_multiple=np.nan).to_csv(out_path, index=False)
        # leere Equity (nur Header) schreiben
        pd.DataFrame({"equity": []}).to_csv(out_eq_path, index=False)
        print(f"[evaluate] Keine gültigen Trades nach SL-Filter. Entfernt: {removed}")
        return

    # 1m Daten laden (für Level-Hit-Simulation – falls R_multiple noch fehlt)
    spx_1m = load_1m_ohlc(spx_path)
    nsx_1m = load_1m_ohlc(nsx_path)

    # falls R_multiple nicht existiert → wie gehabt berechnen
    need_sim = "R_multiple" not in trades.columns or trades["R_multiple"].isna().any()
    if "entry_time_utc" not in trades.columns:
        raise ValueError("Spalte 'entry_time_utc' fehlt in Trades-CSV")
    trades["entry_time_utc"] = pd.to_datetime(trades["entry_time_utc"], utc=True, errors="coerce")

    if need_sim:
        first_hits = []
        r_mults = []
        for _, row in trades.iterrows():
            bias = str(row.get("bias","")).lower().strip()
            symbol = str(row.get("symbol","")).upper()
            entry_ts_utc = row["entry_time_utc"]
            entry = float(row.get("entry_price", np.nan))
            stop  = float(row.get("stop", np.nan))
            tp1   = float(row.get("tp1", np.nan))
            tp2   = float(row.get("tp2", np.nan))
            tp3   = float(row.get("tp3", np.nan))

            if pd.isna(entry_ts_utc) or not np.isfinite(entry) or not np.isfinite(stop):
                first_hits.append("invalid"); r_mults.append(np.nan); continue

            want_long = (bias == "bullish")
            start_ch, end_ch = trade_day_window_ch(entry_ts_utc)
            src = spx_1m if symbol.endswith("SPXUSD") or symbol.startswith("SPX") else nsx_1m
            idx = src.index
            bars = src.loc[(idx >= start_ch) & (idx <= end_ch)].copy()
            if bars.empty:
                first_hits.append("none"); r_mults.append(0.0); continue

            fh, R = compute_partials_R(entry, stop, tp1, tp2, tp3, want_long, bars)
            first_hits.append(fh); r_mults.append(R)

        trades["first_hit"] = first_hits
        trades["R_multiple"] = r_mults

    # Equity-Kurve aus Trades → Kennzahlen
    equity = to_daily_equity_from_trades(trades, start_cash=float(args.start_cash), risk_per_trade=float(args.risk_per_trade))
    metrics = metrics_simple(equity)

    # Equity-Kurve speichern
    pd.DataFrame({"equity": equity}).to_csv(out_eq_path)

    # Kennzahlen als Spalten anhängen und speichern
    for k,v in metrics.items():
        trades[k] = v
    trades.to_csv(out_path, index=False)

    # Kurzreport
    print(f"[evaluate] Trades nach Filter: {len(trades)} | Entfernt (falscher SL): {removed}")
    print(f"[evaluate] Ertrag={metrics['total_return_amount']:.2f}  ({metrics['total_return_pct']*100:.2f}%)")
    print(f"[evaluate] MaxDD={metrics['max_drawdown_amount']:.2f}  ({metrics['max_drawdown_pct']*100:.2f}%)")
    print(f"[evaluate] Tägliche Volatilität={metrics['daily_vol']*100:.2f}%")
    print(f"[evaluate] Ertrag/Risiko (Ertrag / |MaxDD|) = {metrics['return_risk_ratio']:.3f}")

if __name__ == "__main__":
    main()
