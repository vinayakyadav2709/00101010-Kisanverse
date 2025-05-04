import requests
import numpy as np
from datetime import datetime as Datetime
from datetime import timedelta


def get_enhanced_forecast_on_end_date(lat, lon, start_date, end_date, timezone="auto"):
    # Expanded list of available base variables
    variables_base = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
    ]

    url = "https://seasonal-api.open-meteo.com/v1/seasonal"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(variables_base),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception(f"API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    daily = data.get("daily")
    if not daily or "time" not in daily:
        raise KeyError("No 'daily' time series returned.")

    # Find index of the requested end_date
    times = daily["time"]
    if end_date not in times:
        raise ValueError(f"End date {end_date} not in returned time array.")
    idx = times.index(end_date)

    result = {"date": end_date}

    # For each variable, average across ensemble members (filtering out None)
    for var in variables_base:
        member_keys = [k for k in daily.keys() if k.startswith(f"{var}_member")]
        vals = [daily[k][idx] for k in member_keys]
        clean = [v for v in vals if v is not None]
        result[var] = float(np.mean(clean)) if clean else None

    return result


# Example usage
if __name__ == "__main__":
    lat, lon = 19.0760, 72.8777  # Mumbai
    start_date = Datetime.now().strftime("%Y-%m-%d")
    end_date = "2025-07-18"

    forecast = get_enhanced_forecast_on_end_date(lat, lon, start_date, end_date)
    print(forecast)
    # for i in range(30):
    #     end_date = (Datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
    #     forecast = get_enhanced_forecast_on_end_date(
    #         lat, lon, start_date, end_date, timezone="auto"
    #     )
    #     print(f"Day {i + 1}:")
    #     print(f"  Temperature Max: {forecast['temperature_2m_max']}")
    #     print(f"  Temperature Min: {forecast['temperature_2m_min']}")
    #     print(f"  Precipitation Sum: {forecast['precipitation_sum']}")
    #     print(f"  Wind Speed Max: {forecast['wind_speed_10m_max']}")
    #     print(f"  Shortwave Radiation Sum: {forecast['shortwave_radiation_sum']}")
