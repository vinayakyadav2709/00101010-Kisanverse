# filtering.py
import logging
from typing import List, Dict, Optional

# Import directly (flat structure)
import data_sources # Needed to get agronomic data
from models import InputParams, WeatherInfo, AgronomicInfo # Use InputParams

log = logging.getLogger(__name__)

def check_soil_suitability(agronomic_info: AgronomicInfo, soil_type: str) -> bool:
    """Checks if the input soil type is listed as suitable for the crop."""
    if not agronomic_info:
        log.warning("Cannot check soil suitability: AgronomicInfo is missing.")
        return False # Cannot be suitable without info

    # Case-insensitive comparison might be safer
    suitable_soils_lower = [s.lower() for s in agronomic_info.suitable_soils]
    soil_type_lower = soil_type.lower()

    is_suitable = soil_type_lower in suitable_soils_lower
    if not is_suitable:
        log.info(f"[{agronomic_info.crop_name}] Filtered Out: Soil '{soil_type}' not in suitable list: {agronomic_info.suitable_soils}")
    else:
        log.debug(f"[{agronomic_info.crop_name}] Soil check PASSED: Input='{soil_type}', Suitable={agronomic_info.suitable_soils}")
    return is_suitable

def check_water_suitability(agronomic_info: AgronomicInfo, weather_forecast: List[WeatherInfo]) -> bool:
    """
    Performs a basic check if forecasted precipitation aligns somewhat
    with the crop's general water needs category. This is a heuristic.
    Uses the internal WeatherInfo model structure.
    """
    if not agronomic_info:
        log.warning("Cannot check water suitability: AgronomicInfo is missing.")
        return False # Cannot be suitable without info

    water_needs = agronomic_info.water_needs.lower()
    crop_name = agronomic_info.crop_name

    if water_needs == "unknown":
        log.debug(f"[{crop_name}] Water check skipped: Water needs listed as 'Unknown'. Assuming suitable.")
        return True # Assume suitable if needs are unknown

    if not weather_forecast:
        log.warning(f"[{crop_name}] Cannot perform detailed water check: weather_forecast list is empty.")
        # Policy decision: If forecast is missing, maybe eliminate high/very high need crops?
        if water_needs in ["high", "very high"]:
             log.warning(f"[{crop_name}] Filtered Out: Requires '{water_needs}' water, but no forecast data available.")
             return False
        else:
             log.debug(f"[{crop_name}] Water check: Assuming suitable for '{water_needs}' needs without forecast data.")
             return True # Assume low/moderate needs might be met by rainfed conditions

    # Calculate average daily precipitation from the forecast period
    try:
        total_precip = sum(wp.precip for wp in weather_forecast)
        forecast_days = len(weather_forecast)
        avg_daily_precip = total_precip / forecast_days if forecast_days > 0 else 0
    except (AttributeError, TypeError, ZeroDivisionError) as e:
        log.error(f"[{crop_name}] Error calculating average rainfall from forecast: {e}. Check forecast data format. Assuming unsuitable.", exc_info=True)
        return False # Data format issue or empty list after check

    log.debug(f"[{crop_name}] Water check: Needs='{water_needs}', Avg Forecast Rain={avg_daily_precip:.2f}mm/day over {forecast_days} days.")

    # --- Heuristic Thresholds (Adjust based on forecast length and typical season) ---
    # These are simplified examples. Real analysis needs crop stage specific needs.
    # Assuming forecast covers a significant part of early growth (~15-30 days).
    suitable = False
    if water_needs == "low":
        # Low need crops might still need *some* rain, but tolerate drier conditions.
        # Allow suitability even with low predicted rain.
        suitable = avg_daily_precip >= 0.5 # Needs at least minimal rain indication
    elif water_needs == "moderate":
        # Need consistent but not excessive rain.
        suitable = 2.0 <= avg_daily_precip <= 10.0
    elif water_needs == "high":
        # Need significant rainfall.
        suitable = 5.0 <= avg_daily_precip <= 15.0
    elif water_needs == "very high":
        # Need very high rainfall or irrigation. Forecast alone might not be enough.
        suitable = avg_daily_precip >= 10.0 # Threshold for significant rain indication

    if not suitable:
        log.info(f"[{crop_name}] Filtered Out: Forecasted avg rainfall ({avg_daily_precip:.2f}mm/day) seems incompatible with '{water_needs}' water needs based on simple heuristics.")
    else:
         log.debug(f"[{crop_name}] Water check PASSED based on forecast heuristics.")

    return suitable


def filter_agronomically_suitable_crops(potential_crops: List[str], input_data: InputParams) -> List[str]:
    """
    Filters a list of potential crops based on basic agronomic suitability
    (soil type and water needs vs forecast) using data from data_sources.

    XAI Component: Rule-Based System applying domain knowledge checks.
    """
    suitable_crops = []
    log.info(f"--- Running Agronomic Filtering for {len(potential_crops)} potential crops ---")
    log.info(f"Criteria: Soil Type='{input_data.soil_type}', Lat/Lon='{input_data.latitude}/{input_data.longitude}', Forecast Length='{len(input_data.weather_forecast)} days'")

    if not potential_crops:
        log.warning("No potential crops provided to filter.")
        return []

    for crop_name in potential_crops:
        log.debug(f"Checking agronomic suitability for: {crop_name}")
        # Fetch the detailed AgronomicInfo object for the crop
        agronomic_info: Optional[AgronomicInfo] = data_sources.get_crop_agronomic_data(crop_name)

        if not agronomic_info:
            log.warning(f"[{crop_name}] Filtered Out: No detailed agronomic data found.")
            continue

        # --- Check 1: Soil Suitability ---
        soil_ok = check_soil_suitability(agronomic_info, input_data.soil_type)
        if not soil_ok:
            # Logging done within check_soil_suitability
            continue # Move to the next crop

        # --- Check 2: Water Suitability (using forecast) ---
        # This check uses the entire forecast list passed in input_data
        water_ok = check_water_suitability(agronomic_info, input_data.weather_forecast)
        if not water_ok:
             # Logging done within check_water_suitability
             continue # Move to the next crop

        # --- If all checks passed ---
        log.info(f" + '{crop_name}': PASSED basic agronomic checks.")
        suitable_crops.append(crop_name)

    log.info(f"--- Agronomic Filtering Complete ---")
    log.info(f"Found {len(suitable_crops)} agronomically suitable crops out of {len(potential_crops)} potential candidates.")
    log.debug(f"Suitable crops: {suitable_crops}")
    return suitable_crops