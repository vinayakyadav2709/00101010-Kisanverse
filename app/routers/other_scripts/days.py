import json
from datetime import datetime, timedelta
from models import weather


def fetch_and_store_weather_data(
    lat: float, lon: float, output_file: str = "weather_data.json"
):
    """
    Fetches weather data for the next 15 days for the given latitude and longitude
    and stores it in a JSON file.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        output_file (str): Path to the output JSON file.
    """
    try:
        # Calculate the start and end dates
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=15)

        print(
            f"Fetching weather data for Latitude={lat}, Longitude={lon} from {start_date} to {end_date}..."
        )

        # Fetch weather data
        forecast = weather.get_enhanced_forecast_on_end_date(
            lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        print(forecast)
        # Validate forecast is a dictionary
        if not isinstance(forecast, dict):
            raise ValueError(f"Invalid forecast data: {forecast}")

        # Process the data
        weather_data = []
        for day in forecast.get("daily", []):
            date = day.get("date")
            max_temp = day.get("temperature_2m_max")
            min_temp = day.get("temperature_2m_min")
            wind_speed = day.get("wind_speed_10m_max")
            rainfall = day.get("precipitation_sum")

            # Calculate average temperature
            avg_temp = None
            if max_temp is not None and min_temp is not None:
                avg_temp = round((max_temp + min_temp) / 2, 2)

            # Determine rainfall description
            if rainfall is None:
                description_rain = "No data"
            elif rainfall == 0:
                description_rain = "No rain"
            elif rainfall < 5:
                description_rain = "Low"
            elif rainfall < 20:
                description_rain = "Medium"
            else:
                description_rain = "High"

            # Append processed data
            weather_data.append(
                {
                    "date": date,
                    "temperature": avg_temp,
                    "wind_speed_max": round(wind_speed, 2)
                    if wind_speed is not None
                    else None,
                    "rainfall_amount": round(rainfall, 2)
                    if rainfall is not None
                    else None,
                    "description_rain": description_rain,
                }
            )

        # Store the data in a JSON file
        with open(output_file, "w") as f:
            json.dump(weather_data, f, indent=4)

        print(f"Weather data successfully stored in {output_file}")

        # Print the average temperatures
        print("\nAverage Temperatures for the next 15 days:")
        for day in weather_data:
            print(
                f"Date: {day['date']}, Average Temperature: {day['temperature_avg']}°C"
            )

    except Exception as e:
        print(f"Error fetching or storing weather data: {str(e)}")
