
# fetch_weather_lazio.py  ──────────────────────────────────
import asyncio, aiohttp, pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from utils_comuni import get_lazio_comuni

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "rain_sum,snowfall_sum,windspeed_10m_max"
)
HOURLY_VARS = "relativehumidity_2m"

SEM   = asyncio.Semaphore(10)   # max 10 richieste in parallelo
START = datetime(2014, 1, 1)
END   = datetime(2024, 12, 31)

async def fetch(client, row, start, end):
    async with SEM:
        params = {
            "latitude":  row.lat,
            "longitude": row.lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   end.strftime("%Y-%m-%d"),
            "daily": DAILY_VARS,
            "hourly": HOURLY_VARS,
            "timezone": "Europe/Rome",
        }
        async with client.get(BASE_URL, params=params) as r:
            if r.status != 200:
                return pd.DataFrame()  # fallisci silenzioso
            js = await r.json()
            # Daily
            ddf = pd.DataFrame(js["daily"])
            # Humidity: media giornaliera dall'hourly
            hdf = (
                pd.DataFrame(js["hourly"])
                .assign(date=lambda d: d["time"].str.slice(0, 10))
                .groupby("date")["relativehumidity_2m"]
                .mean()
                .reset_index()
            )
            df = ddf.merge(hdf, on="date", how="left")
            df["comune"] = row["Codice Comune"]
            return df

async def main():
    comuni = get_lazio_comuni()
    tasks = []
    async with aiohttp.ClientSession() as client:
        for _, row in comuni.iterrows():
            # loop mensile per bypassare i 31-giorni-limit
            cur = START
            while cur <= END:
                stop = (cur + timedelta(days=30)).replace(day=1) - timedelta(days=1)
                if stop > END:
                    stop = END
                tasks.append(fetch(client, row, cur, stop))
                cur = stop + timedelta(days=1)
        out = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            out.append(await f)
    full = pd.concat(out, ignore_index=True)
    # colonne in ordine carino
    cols = ["date", "comune",
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "rain_sum", "snowfall_sum",
            "windspeed_10m_max", "relativehumidity_2m"]
    full[cols].to_parquet("meteo_lazio_2014_2024.parquet", index=False)
    print("📦 Salvato meteo_lazio_2014_2024.parquet  →", full.shape)

if __name__ == "__main__":
    asyncio.run(main())
