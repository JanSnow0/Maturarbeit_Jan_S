# Maturarbeit_Jan_S
Dieses GitHub Repository enthält slle Dateien, die ich für meine Maturitätsarbeit verwendet habe. Es richtet sich an Personen, die meine schriftliche Arbeit gelesen haben und den dazugehörigen Code, die Resultate oder die verwendeten Marktdaten einsehen möchten.
# Ordnerstruktur
## Code/

Enthält den kompletten Python-Code für das Backtesting:

- **strategy/**
    
    Alle Kernfunktionen der Trading-Strategie. Die genaue Funktionsweise ist in der Maturarbeit erklärt.

- **run_ny.py**

    Startet den gesamten Backtest für die New-York-Session.

- **daily_bias.py**

    Berechnet den täglichen Bias und erstellt die entsprechende CSV-Datei.

- **evaluate_trades.py**

    Wertet die Trades aus und berechnet Ertrag, Volatilität und Drawdown.

- **evaluate_passiv.py**

    Berechnet Ertrag, Volatilität und Drawdown der passiven Investmentstrategie (Buy-and-Hold).

## Equity-Kurven/

Enthält die CSV-Dateien, aus denen die Equity-Kurven der Arbeit berechnet wurden. 

## Trades_Results/

Enthält alle CSV-Dateien, die während der Entwicklung des Codes entstanden sind. 
Im Unterordner **Final Results/** befinden sich die Dateien, die in der finalen Maturarbeit verwendet wurden.

## täglicher Bias/

Die vom Code erzeugten täglichen Bias-Daten (bullish/bearish), gespeichert als CSV.

## Marktdaten.zip

Die verwendeten historischen Marktdaten (1-Minuten-Kerzen, 2023–2024). 
Um die Datei vollständig anzusehen, bitte auf „View raw“ klicken, um sie herunterzuladen.
