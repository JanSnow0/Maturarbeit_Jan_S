# Code/strategy/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Set, Optional
import warnings
import pandas as pd

from .common import load_symbol_csv, CH_TZ
from .news import load_news_table, holidays_set

# ─────────────────────────────────────────────────────────────────────────────
# Generic CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_read_csv(path: Path | str) -> pd.DataFrame:
    """
    Read a CSV with a few sane defaults and clearer errors.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    try:
        df = pd.read_csv(p)
    except Exception as ex:
        raise RuntimeError(f"Failed to read CSV: {p}\n{ex}") from ex
    if df.empty:
        raise ValueError(f"CSV is empty: {p}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Prices / Symbols
# ─────────────────────────────────────────────────────────────────────────────

def load_two_symbols(
    spx_path: Path | str,
    nsx_path: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load SPX and NSX CSVs (Swiss time index, numeric OHLC columns).
    Delegates to common.load_symbol_csv for consistent parsing.
    """
    spx = load_symbol_csv(spx_path)
    nsx = load_symbol_csv(nsx_path)
    return spx, nsx


# ─────────────────────────────────────────────────────────────────────────────
# Daily bias tables
# ─────────────────────────────────────────────────────────────────────────────

def load_bias_csv(path: Path | str) -> pd.DataFrame:
    """
    Load a daily bias CSV with at least columns:
      - 'day' (date or timestamp); becomes YYYY-MM-DD in Europe/Zurich
      - 'bias' (bullish/bearish)
    Optionally:
      - 'scenario' (Int)
    """
    df = safe_read_csv(path)

    if "day" not in df.columns:
        raise ValueError(f"{Path(path).name} must contain 'day' column")
    if "bias" not in df.columns:
        raise ValueError(f"{Path(path).name} must contain 'bias' column (bullish/bearish)")

    # Normalize day to CH date string
    day_ts = pd.to_datetime(df["day"], errors="coerce", utc=True)
    if day_ts.isna().all():
        # Try parsing as naive local strings (fallback)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            day_ts = pd.to_datetime(df["day"], errors="coerce", utc=False).dt.tz_localize(CH_TZ)
    day_ch = day_ts.dt.tz_convert(CH_TZ).dt.strftime("%Y-%m-%d")

    out_cols = ["day", "bias"]
    out = pd.DataFrame({
        "day": day_ch,
        "bias": df["bias"].astype(str).str.lower().str.strip(),
    })

    if "scenario" in df.columns:
        out["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
        out_cols.append("scenario")

    return out[out_cols]


# ─────────────────────────────────────────────────────────────────────────────
# News + Holidays
# ─────────────────────────────────────────────────────────────────────────────

def load_news_and_holidays(
    news_path: Optional[Path | str],
    holidays: Optional[str | Path] = "use_news",
) -> Tuple[Optional[pd.DataFrame], Set[str]]:
    """
    Load the news table (converted to Europe/Zurich) and derive a set of
    holiday days (YYYY-MM-DD).

    holidays parameter:
      - "use_news": infer bank holidays from the news table (title contains "Bank Holiday")
      - None: no holidays filtering
      - Path to a CSV containing a 'day' or 'date' column (will be parsed to CH date)
    """
    # News
    news_df = load_news_table(Path(news_path)) if news_path else None

    # Holidays
    if holidays == "use_news":
        holi = holidays_set(news_df) if news_df is not None else set()
    elif holidays is None:
        holi = set()
    else:
        hdf = safe_read_csv(holidays)
        for c in ("day", "date", "Date"):
            if c in hdf.columns:
                ts = pd.to_datetime(hdf[c], errors="coerce", utc=True)
                holi = set(ts.dt.tz_convert(CH_TZ).dt.strftime("%Y-%m-%d").dropna().tolist())
                break
        else:
            raise ValueError("Holidays CSV must include a 'day' or 'date' column.")
    return news_df, holi
