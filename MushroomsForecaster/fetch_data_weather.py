#!/usr/bin/env python3
"""
data_pipeline.py --> must update API

Pipeline to fetch macrofungi observations from iNaturalist and pair them with historical
weather covariates from Open‑Meteo. Outputs clean, analysis‑ready CSV files that can be
incrementally extended.

Usage (initial backfill):
    python data_pipeline.py --bbox "41.8,12.2,42.1,12.8" --start 2015‑01‑01 --end 2025‑05‑29

Usage (daily cron incremental):
    python data_pipeline.py --bbox "41.8,12.2,42.1,12.8" --incremental

Dependencies:
    pip install aiohttp pandas tqdm python-dateutil geopy pyproj

Author: Your Name, 2025-05-29
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os, json, argparse
from datetime import datetime, timedelta, date
from dateutil import parser as dateparser
from pathlib import Path
from typing import List, Dict, Any, Tuple

INAT_ENDPOINT = "https://api.inaturalist.org/v1/observations"
OPEN_METEO_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
MAX_CONCURRENT_REQ = 10
INAT_PAGE_SIZE = 200  # API maximum
OUTPUT_DIR = Path("data")

def parse_bbox(bbox: str) -> Tuple[float,float,float,float]:
    lat1,lon1,lat2,lon2 = map(float, bbox.split(","))
    return lat1,lon1,lat2,lon2

async def fetch_json(session: aiohttp.ClientSession, url:str, params:Dict[str,Any]) -> Dict[str,Any]:
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        resp.raise_for_status()
        return await resp.json()

async def fetch_inat_page(session, page:int, params_base:Dict[str,Any]) -> List[Dict[str,Any]]:
    params = params_base.copy()
    params.update({"page": page, "per_page": INAT_PAGE_SIZE})
    data = await fetch_json(session, INAT_ENDPOINT, params)
    return data.get("results", [])

async def fetch_inat_observations(start_date:str, end_date:str, bbox:str, taxon_id:int=47170) -> pd.DataFrame:
    """
    Downloads macrofungi observations (taxon_id default 47170) for date range and bbox.
    """
    lat1,lon1,lat2,lon2 = parse_bbox(bbox)
    params_base = {
        "iconic_taxa": "Fungi",
        "taxon_id": taxon_id,
        "d1": start_date,
        "d2": end_date,
        "nelat": lat2, "nelng": lon2,
        "swlat": lat1, "swlng": lon1,
        "order": "asc",
        "order_by": "observed_on",
        "ttl": 0
    }
    observations = []
    async with aiohttp.ClientSession() as session:
        first = await fetch_json(session, INAT_ENDPOINT, {**params_base, "page":1, "per_page":1})
        total_results = first.get("total_results", 0)
        total_pages = int(np.ceil(total_results / INAT_PAGE_SIZE))
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQ)

        async def task(p):
            async with sem:
                return await fetch_inat_page(session, p, params_base)

        pages = await asyncio.gather(*[task(p) for p in range(1, total_pages+1)])
        for page in pages:
            observations.extend(page)

    df = pd.json_normalize(observations)
    return df

async def fetch_open_meteo_history(lat:float, lon:float, start_date:str, end_date:str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "UTC"
    }
    async with aiohttp.ClientSession() as session:
        data = await fetch_json(session, OPEN_METEO_ENDPOINT, params)
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["lat"] = lat
    df["lon"] = lon
    return df

async def augment_observations_with_weather(obs_df:pd.DataFrame) -> pd.DataFrame:
    obs_df["lat_round"] = obs_df["geojson.coordinates"].apply(lambda x: round(x[1],2))
    obs_df["lon_round"] = obs_df["geojson.coordinates"].apply(lambda x: round(x[0],2))
    obs_df["date"] = pd.to_datetime(obs_df["observed_on"], utc=True).dt.date

    unique_keys = obs_df[["lat_round","lon_round","date"]].drop_duplicates()

    async def fetch_and_process(lat,lon,d):
        start = str(d - timedelta(days=30))
        end = str(d)
        wdf = await fetch_open_meteo_history(lat, lon, start, end)
        if wdf.empty:
            return None
        wdf["date"] = wdf["time"].dt.date
        agg = (wdf.groupby("date")
                   .agg(temp_mean=("temperature_2m","mean"),
                        humid_mean=("relative_humidity_2m","mean"),
                        rain_sum=("precipitation","sum"))
                   .reset_index())
        win = agg[agg["date"]<=d]
        return {
            "lat_round": lat,
            "lon_round": lon,
            "date": d,
            "temp_mean7": win.tail(7)["temp_mean"].mean(),
            "hum_mean7": win.tail(7)["humid_mean"].mean(),
            "rain_sum7": win.tail(7)["rain_sum"].sum(),
            "temp_mean30": win.tail(30)["temp_mean"].mean(),
            "rain_sum30": win.tail(30)["rain_sum"].sum()
        }

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQ)
    async def limited_task(row):
        lat,lon,d = row
        async with sem:
            return await fetch_and_process(lat,lon,d)

    tasks = [limited_task(tuple(r)) for r in unique_keys.itertuples(index=False)]
    features = await asyncio.gather(*tasks)
    feat_df = pd.DataFrame([f for f in features if f])
    merged = (obs_df
              .merge(feat_df, on=["lat_round","lon_round","date"], how="left")
              .drop(columns=["lat_round","lon_round"]))
    return merged

def incremental_dates(meta:Path, start:date, end:date) -> Tuple[str,str]:
    last_end = None
    if meta.exists():
        with open(meta) as f:
            last_end = dateparser.parse(json.load(f)["last_end"]).date()
    actual_start = (last_end + timedelta(days=1)) if last_end else start
    return str(actual_start), str(end)

def save_metadata(meta:Path, end:str):
    meta.parent.mkdir(exist_ok=True, parents=True)
    with open(meta,"w") as f:
        json.dump({"last_end": end}, f)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", required=True, help="lat1,lon1,lat2,lon2")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--outfile", default="observations_weather.csv")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUTPUT_DIR / "meta.json"

    if args.incremental:
        start_date, end_date = incremental_dates(meta_path, dateparser.parse(args.start).date(), dateparser.parse(args.end).date())
    else:
        start_date, end_date = args.start, args.end

    print(f"Fetching observations {start_date} to {end_date}...")
    obs_df = await fetch_inat_observations(start_date, end_date, args.bbox)
    if obs_df.empty:
        print("No new observations.")
        return

    merged_df = await augment_observations_with_weather(obs_df)

    outfile = OUTPUT_DIR / args.outfile
    mode = "a" if outfile.exists() else "w"
    header = not outfile.exists()
    merged_df.to_csv(outfile, index=False, mode=mode, header=header)

    save_metadata(meta_path, end_date)
    print(f"Appended {len(merged_df)} rows to {outfile}")

if __name__ == "__main__":
    asyncio.run(main())