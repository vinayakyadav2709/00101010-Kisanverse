import json
from datetime import datetime, timedelta
from models import weather


def get_rainfall_description(rainfall):
    """
    Categorize rainfall into low, medium, or high based on its value.
    """
    if rainfall == 0:
        return "none"
    elif rainfall < 2.5:
        return "low"
    elif rainfall < 7.6:
        return "medium"
    else:
        return "high"


def fetch_weather_data_for_coordinate(latitude, longitude):
    try:
        # Start date as today
        start_date = datetime.now().date()
        print(f"Fetching weather data for Latitude={latitude}, Longitude={longitude}")
        print(f"Start date: {start_date}")

        # Prepare to fetch data for the next 16 days
        weather_summary = []
        for day_offset in range(16):
            # Calculate the date for each day
            current_date = start_date + timedelta(days=day_offset)
            print(f"Fetching weather data for date: {current_date}")

            # Fetch weather data
            try:
                forecast = weather.get_enhanced_forecast_on_end_date(
                    latitude,
                    longitude,
                    current_date.strftime("%Y-%m-%d"),
                    current_date.strftime("%Y-%m-%d"),
                )
                # Validate forecast is a dictionary
                if not isinstance(forecast, dict):
                    raise ValueError(f"Invalid forecast data: {forecast}")

                # Extract required fields
                date = current_date.strftime("%Y-%m-%d")
                rainfall = round(forecast.get("precipitation_sum", 0), 2)
                rainfall_description = get_rainfall_description(rainfall)
                wind_speed = round(forecast.get("wind_speed_10m_max", 0), 2)
                temperature = round(
                    (
                        forecast.get("temperature_2m_max", 0)
                        + forecast.get("temperature_2m_min", 0)
                    )
                    / 2,
                    2,
                )

                # Append the summarized data
                weather_summary.append(
                    {
                        "date": date,
                        "rainfall": rainfall,
                        "rainfall_description": rainfall_description,
                        "wind_speed": wind_speed,
                        "temperature": temperature,
                    }
                )
                print(f"Successfully fetched weather data for {current_date}.")
            except Exception as e:
                print(
                    f"Error fetching forecast for Latitude={latitude}, Longitude={longitude} on {current_date}: {e}"
                )
                continue

        # Save the weather summary to a JSON file
        output_file = f"weather.json"
        with open(output_file, "w") as json_file:
            json.dump(weather_summary, json_file, indent=4)
        print(f"Weather summary saved to {output_file}")

    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")


# Run the function
if __name__ == "__main__":
    # Replace with the desired latitude and longitude
    latitude = 28.7041  # Example: Latitude for Delhi
    longitude = 77.1025  # Example: Longitude for Delhi
    fetch_weather_data_for_coordinate(latitude, longitude)
