# Code/evaluate_passiv.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CH_TZ = "Europe/Zurich"

def load_ohlc_any(path: Path) -> pd.DataFrame:
    """
    Robuster Loader für 1m/Mehrfrequenz-CSV:
    - sucht Zeitspalte (ts/time/timestamp/datetime/date)
    - konvertiert nach CH-Zeit
    - normalisiert OHLC auf ['open','high','low','close']
    """
    df = pd.read_csv(path)
    ts_candidates = [c for c in df.columns if c.lower() in ("ts","time","timestamp","datetime","date")]
    if not ts_candidates:
        raise ValueError(f"Keine Zeitspalte in {path}")
    ts = pd.to_datetime(df[ts_candidates[0]], utc=True, errors="coerce").dt.tz_convert(CH_TZ)
    df = df.set_index(ts).sort_index()

    # OHLC-Spalten finden
    cands = [
        ("open","high","low","close"),
        ("open_1m","high_1m","low_1m","close_1m"),
        ("o","h","l","c"),
        ("Open","High","Low","Close"),
    ]
    found = None
    for o,h,l,c in cands:
        if all(col in df.columns for col in (o,h,l,c)):
            found = (o,h,l,c)
            break
    if found is None:
        raise ValueError(f"Keine passenden OHLC-Spalten in {path}")
    o,h,l,c = found

    out = df[[o,h,l,c]].copy()
    out.columns = ["open","high","low","close"]
    for col in ["open","high","low","close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def daily_close(ch_df_ohlc: pd.DataFrame) -> pd.Series:
    """
    Tages-Schlusskurs-Serie (letzte verfügbare Kerze je Tag, CH-Zeit).
    """
    if ch_df_ohlc.index.tz is None:
        ch_df_ohlc = ch_df_ohlc.tz_localize(CH_TZ)
    return ch_df_ohlc["close"].resample("1D").last().dropna()

def metrics_simple(equity: pd.Series) -> dict:
    """
    Kennzahlen:
      - total_return_amount, total_return_pct
      - max_drawdown_amount, max_drawdown_pct
      - daily_vol (Stdabw. der täglichen Renditen, nicht annualisiert)
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

    # tägliche (nicht annualisierte) Volatilität
    rets = equity.pct_change().dropna()
    daily_vol = float(rets.std(ddof=1)) if len(rets) > 1 else np.nan

    # Max Drawdown (Betrag & %)
    roll_max = equity.cummax()
    dd_amt_series = roll_max - equity
    max_dd_amount = float(dd_amt_series.max()) if len(dd_amt_series) else 0.0

    dd_pct_series = equity / roll_max - 1.0
    max_dd_pct    = float(dd_pct_series.min()) if len(dd_pct_series) else 0.0  # negativ

    # Ertrag/Risiko
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

def main():
    ap = argparse.ArgumentParser(description="Passive 50/50 Buy&Hold Evaluator (SPX, NSX) — einfache Kennzahlen")
    ap.add_argument("--spx", required=True, help="SPX CSV (1m/5m/1h; 1m close wird genutzt)")
    ap.add_argument("--nsx", required=True, help="NSX CSV (1m/5m/1h; 1m close wird genutzt)")
    ap.add_argument("--start_cash", type=float, default=10000.0, help="Startkapital (Default 10000)")
    ap.add_argument("--out_equity", required=True, help="Output CSV der Portfolio-Equity-Kurve (täglich)")
    ap.add_argument("--out_summary", required=True, help="Output CSV mit Kompakt-Kennzahlen")
    args = ap.parse_args()

    spx = load_ohlc_any(Path(args.spx))
    nsx = load_ohlc_any(Path(args.nsx))

    spx_d = daily_close(spx)
    nsx_d = daily_close(nsx)

    # gemeinsame Tage
    idx = spx_d.index.intersection(nsx_d.index)
    spx_d = spx_d.loc[idx]
    nsx_d = nsx_d.loc[idx]
    if len(idx) < 2:
        raise ValueError("Zu wenige gemeinsame Tage zwischen SPX und NSX für die passive Auswertung.")

    # Buy&Hold: 50/50 am ersten gemeinsamen Tag
    start_val = float(args.start_cash)
    alloc_spx = start_val * 0.5
    alloc_nsx = start_val * 0.5

    spx_shares = alloc_spx / float(spx_d.iloc[0])
    nsx_shares = alloc_nsx / float(nsx_d.iloc[0])

    equity = spx_shares * spx_d + nsx_shares * nsx_d
    equity.name = "equity"

    # Kennzahlen
    m = metrics_simple(equity)

    # Speichern
    pd.DataFrame({"equity": equity}).to_csv(Path(args.out_equity))
    pd.DataFrame([m]).to_csv(Path(args.out_summary), index=False)

    # Kurzreport
    print(f"Passive 50/50 — Start: {equity.index[0].date()}  Ende: {equity.index[-1].date()}")
    print(f"Start={m['start_value']:.2f}  Ende={m['end_value']:.2f}  Ertrag={m['total_return_amount']:.2f}  ({m['total_return_pct']*100:.2f}%)")
    print(f"MaxDD={m['max_drawdown_amount']:.2f}  ({m['max_drawdown_pct']*100:.2f}%)  Tägliche Vol={m['daily_vol']*100:.2f}%")
    print(f"Ertrag/Risiko (Ertrag / |MaxDD|) = {m['return_risk_ratio']:.3f}")

if __name__ == "__main__":
    main()
