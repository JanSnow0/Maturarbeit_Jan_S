from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd

CH_TZ = "Europe/Zurich"

def load_news_table(path: Path | None) -> pd.DataFrame | None:
    """
    Expect a CSV with at least:
      - a timestamp column (timestamp/Time/Date/etc.)
      - a text/title column describing the event
    We convert timestamps to Europe/Zurich.
    """
    if path is None:
        return None
    df = pd.read_csv(path)

    tcol = next((c for c in ["timestamp","time","Time","datetime","Date","date"] if c in df.columns), None)
    if tcol is None:
        raise ValueError("news csv must include a time column")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(CH_TZ)

    df = df.drop(columns=[tcol])
    df.insert(0, "timestamp", ts)

    if "title" in df.columns:
        pass
    elif "event" in df.columns:
        df = df.rename(columns={"event":"title"})
    elif "text" in df.columns:
        df = df.rename(columns={"text":"title"})
    else:
        # fall back to first object column or create one
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if obj_cols:
            df = df.rename(columns={obj_cols[0]:"title"})
        else:
            df["title"] = ""

    return df

def holidays_set(news_df: pd.DataFrame | None) -> set[str]:
    if news_df is None or news_df.empty:
        return set()
    txtcol = "title" if "title" in news_df.columns else None
    if txtcol is None:
        return set()
    mask = news_df[txtcol].astype(str).str.contains("Bank Holiday", case=False, na=False)
    days = news_df.loc[mask, "timestamp"].dt.strftime("%Y-%m-%d")
    return set(days.tolist())
