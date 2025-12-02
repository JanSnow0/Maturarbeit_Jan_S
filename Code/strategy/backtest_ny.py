from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import time

from .news import load_news_table, holidays_set
from .common import load_symbol_csv
from .recipes import Scenario1NY, Scenario2NY, Scenario3NY, RiskConfig

def _pf(msg: str):
    print(msg, flush=True)

def _parse_scenarios(scenarios) -> tuple[int, ...]:
    if scenarios is None:
        return (1,2,3)
    if isinstance(scenarios, (list, tuple)):
        return tuple(int(x) for x in scenarios)
    return tuple(int(x.strip()) for x in str(scenarios).split(",") if x.strip())

def _load_bias_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "day" not in df.columns:
        raise ValueError(f"{path.name} must contain 'day' column")
    if "bias" not in df.columns:
        raise ValueError(f"{path.name} must contain 'bias' column (bullish/bearish)")
    df["day"] = pd.to_datetime(df["day"], errors="coerce", utc=True)\
                   .dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d")
    if "scenario" in df.columns:
        df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
    return df[["day","bias"] + (["scenario"] if "scenario" in df.columns else [])]

def run_ny_engine(
    spx_file: Path,
    nsx_file: Path,
    bias_spx_file: Path,
    bias_nsx_file: Path,
    news_file: Path | None = None,
    holidays: str | Path | None = None,
    out_trades: Path | None = None,
    scenarios: tuple[int, ...] | str | None = (1,2,3),
    debug_path: Path | None = None,
    max_seconds_per_day: float = 5.0,
) -> pd.DataFrame:

    _pf("▶ Starting NY backtest…")
    scenarios = _parse_scenarios(scenarios)
    _pf(f"  - Scenarios: {scenarios}")

    _pf(f"  - Loading SPX: {spx_file}")
    spx = load_symbol_csv(spx_file)
    _pf(f"    SPX rows: {len(spx):,}")

    _pf(f"  - Loading NSX: {nsx_file}")
    nsx = load_symbol_csv(nsx_file)
    _pf(f"    NSX rows: {len(nsx):,}")

    _pf(f"  - Loading bias SPX: {bias_spx_file}")
    bias_spx = _load_bias_csv(bias_spx_file)
    _pf(f"    bias SPX days: {len(bias_spx):,}")

    _pf(f"  - Loading bias NSX: {bias_nsx_file}")
    bias_nsx = _load_bias_csv(bias_nsx_file)
    _pf(f"    bias NSX days: {len(bias_nsx):,}")

    if news_file:
        _pf(f"  - Loading news: {news_file}")
    news = load_news_table(news_file) if news_file else None
    if news is not None:
        _pf(f"    news rows: {len(news):,}")

    if holidays == "use_news":
        holi = holidays_set(news) if news is not None else set()
        _pf(f"  - Holidays from news: {len(holi)} days")
    elif holidays is None:
        holi = set()
        _pf("  - Holidays: none")
    else:
        _pf(f"  - Loading holidays CSV: {holidays}")
        h = pd.read_csv(holidays)
        for c in ["day","date","Date"]:
            if c in h.columns:
                holi = set(pd.to_datetime(h[c], errors="coerce", utc=True)
                             .dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d"))
                break
        else:
            raise ValueError("Holidays CSV must include a 'day' or 'date' column.")
        _pf(f"    holidays days: {len(holi)}")

    risk = RiskConfig()
    s1 = Scenario1NY(risk); s2 = Scenario2NY(risk); s3 = Scenario3NY(risk)

    out_rows: list[dict] = []
    debug_rows: list[dict] = []

    total_days = len(bias_spx)
    _pf(f"▶ Iterating days: {total_days}…")
    for i, (_, row_spx) in enumerate(bias_spx.iterrows(), start=1):
        t0 = time.time()
        day = str(row_spx["day"])
        if i % 5 == 0 or i == 1:
            _pf(f"  - Day {i}/{total_days}: {day}")

        if day in holi:
            debug_rows.append({"symbol":"SPXUSD","day":day,"recipe":"holiday (skipped)"})
            continue

        row_nsx = bias_nsx.loc[bias_nsx["day"] == day]
        if row_nsx.empty:
            debug_rows.append({"symbol":"NSXUSD","day":day,"recipe":"missing NSX bias"})
            continue

        trade_today: list[dict] = []
        for scen in scenarios:
            if time.time() - t0 > max_seconds_per_day:
                _pf(f"    ⏱ timeout on {day} — skipped")
                debug_rows.append({"symbol":"SPXUSD","day":day,"recipe":"timeout"})
                break
            _pf(f"    • evaluating S{scen}…")
            try:
                if scen == 1:
                    trades = s1.run_day("SPXUSD","NSXUSD", spx, nsx, spx, nsx, row_spx)
                elif scen == 2:
                    trades = s2.run_day("SPXUSD","NSXUSD", spx, nsx, spx, nsx, row_spx)
                else:
                    trades = s3.run_day("SPXUSD","NSXUSD", spx, nsx, spx, nsx, row_spx)
            except Exception as ex:
                _pf(f"      ! S{scen} error: {ex}")
                trades = []
            if trades:
                _pf(f"      ✔ Trade on {day} via S{scen}: {trades[0].get('symbol')} entry={trades[0].get('entry_time_utc')}")
                trade_today = trades
                break

        if trade_today:
            out_rows.extend(trade_today)
            debug_rows.extend(trade_today)
        else:
            if not debug_rows or debug_rows[-1].get("day") != day:
                debug_rows.append({"symbol":"SPXUSD","day":day,"recipe":"no-trade"})

    out = pd.DataFrame(out_rows)
    _pf(f"▶ Done. Trades: {len(out):,}")

    if out_trades is not None:
        Path(out_trades).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_trades, index=False)
        _pf(f"  - Saved trades → {out_trades}")

    if debug_path is not None:
        dbg = pd.DataFrame(debug_rows)
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        dbg.to_csv(debug_path, index=False)
        _pf(f"  - Saved debug  → {debug_path}")

    return out
