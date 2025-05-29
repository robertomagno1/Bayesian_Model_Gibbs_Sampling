# utils_comuni.py

import pandas as pd

ISTAT_CSV = (
    "https://www.istat.it/storage/regioni/Elenco-comuni-italiani.csv"
)

def get_lazio_comuni() -> pd.DataFrame:
    df = pd.read_csv(ISTAT_CSV, sep=";", dtype=str)
    lazio = df[df["Codice Regione"] == "12"].copy()
    lazio["lat"] = lazio["Latitudine"].astype(float)
    lazio["lon"] = lazio["Longitudine"].astype(float)
    return lazio[["Codice Comune", "Denominazione (Italiana)", "lat", "lon"]]
