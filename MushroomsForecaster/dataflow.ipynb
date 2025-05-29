import requests
import pandas as pd
from datetime import datetime

def fetch_weather(lat, lon, start, end, outfile):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max"],
        "timezone": "Europe/Rome"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Errore nella richiesta:", response.status_code)
        return
    
    data = response.json()
    df = pd.DataFrame(data['daily'])
    df.to_csv(outfile, index=False)
    print(f"Dati salvati in {outfile}")

# ESEMPIO USO
if __name__ == "__main__":
    fetch_weather(
        lat=41.9028,
        lon=12.4964,
        start="2024-01-01",
        end="2024-12-31",
        outfile="meteo_roma_2024.csv"
    )
