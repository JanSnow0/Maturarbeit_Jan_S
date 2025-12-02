# Code/run_ny.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# Strategy-Module
from strategy.loaders import load_symbol_csv
from strategy.recipes import Scenario1NY, Scenario2NY, Scenario3NY, RiskConfig
from strategy.news import load_news_table, holidays_set

# SMT zur optionalen Deaktivierung
import strategy.smt as smt

CH_TZ = "Europe/Zurich"

def parse_args():
    ap = argparse.ArgumentParser(description="NY Backtest Runner (mit --no-smt Option)")
    ap.add_argument("--spx", required=True, help="Pfad SPX CSV (1m/5m/1h in CH-Zeit)")
    ap.add_argument("--nsx", required=True, help="Pfad NSX CSV (1m/5m/1h in CH-Zeit)")
    ap.add_argument("--bias_spx", required=True, help="Pfad daily_bias_SPXUSD.csv")
    ap.add_argument("--bias_nsx", required=True, help="Pfad daily_bias_NSXUSD.csv")
    ap.add_argument("--news", required=False, default="", help="Pfad News/Feiertage CSV (CH Zeit empfohlen)")
    ap.add_argument("--holidays", required=False, default="none",
                    choices=["none","use_news"],
                    help="Feiertage ausschließen: none|use_news (aus news.csv)")
    ap.add_argument("--scenarios", required=False, default="1,2,3",
                    help="Kommagetrennt: 1,2,3")
    ap.add_argument("--out", required=True, help="Output: Trades CSV")
    ap.add_argument("--debug", required=False, default="", help="Optional: Debug CSV")
    ap.add_argument("--no-smt", action="store_true",
                    help="SMT komplett deaktivieren (erzwingt 50/50-Fallback in recipes.py)")
    return ap.parse_args()

def _disable_smt():
    # Überschreibt die SMT-Symbolwahl → (None, False) == kein SMT → recipes.py macht 50/50
    def _noop_choose_symbol_for_day(*args, **kwargs):
        return (None, False)
    smt.choose_symbol_for_day = _noop_choose_symbol_for_day
    print("⚠ SMT deaktiviert: choose_symbol_for_day → (None, False)")

def load_bias_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Erwartete Spalten: day, bias[, scenario] – wir normalisieren
    if "day" not in df.columns:
        # Fallback: versuchen Timestamp zu parsen
        for c in ["date","Date","day_str","Day","DAY"]:
            if c in df.columns:
                df["day"] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None).dt.date.astype(str)
                break
    # Bias lower
    if "bias" in df.columns:
        df["bias"] = df["bias"].astype(str).str.lower().str.strip()
    # Sicherstellen, dass day String ist
    df["day"] = df["day"].astype(str)
    # optional Szenario integer/str – wird in recipes nicht zwingend benötigt
    if "scenario" in df.columns:
        try:
            df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass
    return df[["day","bias"] + ([ "scenario"] if "scenario" in df.columns else [])].drop_duplicates()

def main():
    args = parse_args()

    # SMT optional abschalten
    if args.no_smt:
        _disable_smt()

    # Laden Symboldaten
    spx_file = Path(args.spx)
    nsx_file = Path(args.nsx)
    print(f"  - Lade SPX: {spx_file}")
    spx = load_symbol_csv(spx_file)
    print(f"    SPX Zeilen: {len(spx)}")
    print(f"  - Lade NSX: {nsx_file}")
    nsx = load_symbol_csv(nsx_file)
    print(f"    NSX Zeilen: {len(nsx)}")

    # Bias laden
    bias_spx_path = Path(args.bias_spx)
    bias_nsx_path = Path(args.bias_nsx)
    print(f"  - Lade Bias SPX: {bias_spx_path}")
    bias_spx = load_bias_csv(bias_spx_path)
    print(f"    Bias SPX Tage: {len(bias_spx)}")
    print(f"  - Lade Bias NSX: {bias_nsx_path}")
    bias_nsx = load_bias_csv(bias_nsx_path)
    print(f"    Bias NSX Tage: {len(bias_nsx)}")

    # News/Holidays
    holi_set = set()
    if args.holidays == "use_news" and args.news:
        news_path = Path(args.news)
        print(f"  - Lade News: {news_path}")
        news = load_news_table(news_path)
        print(f"    News-Zeilen: {len(news)}")
        holi_set = holidays_set(news)
        print(f"    Feiertage aus News: {len(holi_set)}")
    else:
        print("  - Keine Feiertagsfilter aktiv.")

    # Szenarienliste
    scen_str = args.scenarios.strip()
    scen_list = []
    if scen_str:
        for tok in scen_str.split(","):
            tok = tok.strip()
            if tok in ("1","2","3"):
                scen_list.append(int(tok))
    if not scen_list:
        scen_list = [1,2,3]
    print(f"▶ Szenarien aktiv: {scen_list}")

    # Szenario-Instanzen
    risk = RiskConfig()
    S1 = Scenario1NY(risk)
    S2 = Scenario2NY(risk)
    S3 = Scenario3NY(risk)

    # Wir treiben den Loop über die SPX-Bias-Tage (NSX wird intern benutzt)
    days = sorted(bias_spx["day"].unique())
    print(f"▶ Iteriere Tage: {len(days)}…")

    out_rows = []
    for i, day in enumerate(days, start=1):
        # Skip Holidays
        if day in holi_set:
            continue

        # Bias ROW (SPX maßgeblich)
        row_spx = bias_spx.loc[bias_spx["day"] == day]
        if row_spx.empty:
            continue
        row_spx = row_spx.iloc[0]

        # Für Log
        if i % 20 == 1 or i == 1:
            print(f"  - Tag {i}/{len(days)}: {day}")

        # Je Szenario versuchen
        if 1 in scen_list:
            try:
                t1 = S1.run_day("SPXUSD","NSXUSD", spx, nsx, spx, nsx, row_spx)
                out_rows.extend(t1)
            except Exception as e:
                print(f"    [S1 Fehler {day}]: {e}", file=sys.stderr)

        if 2 in scen_list:
            try:
                t2 = S2.run_day("SPXUSD","NSXUSD", spx, nsx, spx, nsx, row_spx)
                out_rows.extend(t2)
            except Exception as e:
                print(f"    [S2 Fehler {day}]: {e}", file=sys.stderr)

        if 3 in scen_list:
            try:
                t3 = S3.run_day("SPXUSD","NSXUSD", spx, nsx, spx, nsx, row_spx)
                out_rows.extend(t3)
            except Exception as e:
                print(f"    [S3 Fehler {day}]: {e}", file=sys.stderr)

    # Output schreiben
    out_path = Path(args.out)
    if out_rows:
        df_out = pd.DataFrame(out_rows)
        df_out.to_csv(out_path, index=False)
        print(f"✓ Fertig. Trades: {len(df_out)} → {out_path}")
    else:
        # leere Datei mit Headern
        cols = ["symbol","day","scenario","bias","recipe","entry_time_utc","entry_price","stop","tp1","tp2","tp3"]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        print(f"✓ Fertig. Keine Trades → {out_path}")

    # Optional Debug
    if args.debug:
        dbg_path = Path(args.debug)
        try:
            # Minimaler Debug-Drop: nur Tage-Liste
            pd.DataFrame({"day": days}).to_csv(dbg_path, index=False)
            print(f"  (Debug geschrieben: {dbg_path})")
        except Exception as e:
            print(f"  (Debug konnte nicht geschrieben werden: {e})", file=sys.stderr)

if __name__ == "__main__":
    main()
