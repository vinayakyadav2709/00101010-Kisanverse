# filtering.py
import logging
from typing import List, Dict, Optional

# Assuming these imports remain necessary for your project structure
import models.llm.data_sources as data_sources
from models.llm.models import InputParams, WeatherInfo, AgronomicInfo

log = logging.getLogger(__name__)


def check_soil_suitability(agronomic_info: AgronomicInfo, soil_type: str) -> bool:
    """
    Checks if the input soil type (expected UPPERCASE) is listed
    as suitable for the crop (suitable_soils list also expected UPPERCASE).
    """
    if not agronomic_info:
        log.warning("Cannot check soil suitability: AgronomicInfo is missing.")
        return False

    # Direct, case-sensitive comparison (assuming uppercase inputs)
    suitable_soils = agronomic_info.suitable_soils
    is_suitable = soil_type in suitable_soils  # Simple check now

    if not is_suitable:
        log.info(
            f"[{agronomic_info.crop_name}] Filtered Out: Soil '{soil_type}' not in suitable list: {suitable_soils}"
        )
    else:
        log.debug(
            f"[{agronomic_info.crop_name}] Soil check PASSED: Input='{soil_type}', Suitable={suitable_soils}"
        )
    return is_suitable


# Function name reverted to original, but logic remains lenient
def check_water_suitability(
    agronomic_info: AgronomicInfo, weather_forecast: List[WeatherInfo]
) -> bool:
    """
    Performs a check if forecasted precipitation seems grossly
    inadequate for the crop's general water needs (Lenient Logic).
    Uses the internal WeatherInfo model structure.
    Focuses on minimum plausible rainfall.
    """
    if not agronomic_info:
        log.warning("Cannot check water suitability: AgronomicInfo is missing.")
        return False  # Cannot be suitable without info

    # Handle variations like "Moderate (...)" - take the first word
    water_needs_raw = agronomic_info.water_needs or "Unknown"
    water_needs = water_needs_raw.split()[0].lower()  # Get 'low', 'moderate', etc.
    crop_name = agronomic_info.crop_name

    if water_needs == "unknown":
        log.debug(
            f"[{crop_name}] Water check skipped: Water needs listed as 'Unknown' or unparseable ('{water_needs_raw}'). Assuming suitable."
        )
        return True  # Lenient: assume suitable if needs unknown

    if not weather_forecast:
        log.warning(
            f"[{crop_name}] Cannot perform water check: weather_forecast list is empty. Assuming suitable due to lack of data."
        )
        # Lenient policy: Don't filter if forecast is missing
        return True

    # Calculate average daily precipitation
    try:
        total_precip = sum(
            wp.precip for wp in weather_forecast if wp.precip is not None
        )  # Handle potential None values
        forecast_days = len(weather_forecast)
        avg_daily_precip = total_precip / forecast_days if forecast_days > 0 else 0
    except (AttributeError, TypeError, ZeroDivisionError) as e:
        log.error(
            f"[{crop_name}] Error calculating average rainfall from forecast: {e}. Check forecast data format. Assuming suitable due to error.",
            exc_info=True,  # Log traceback for debugging
        )
        # Lenient policy: Don't filter if calculation fails
        return True

    log.debug(
        f"[{crop_name}] Water check: Needs='{water_needs}', Avg Forecast Rain={avg_daily_precip:.2f}mm/day over {forecast_days} days."
    )

    # --- Lenient Heuristic Thresholds (Focus on Minimums) ---
    # Only filter if the forecast shows *very little* rain for higher need crops.
    suitable = True  # Assume suitable by default
    if water_needs == "low":
        # Low need crops are generally drought-tolerant, suitable even with minimal rain.
        # No specific check needed in lenient mode.
        pass  # Suitable
    elif water_needs == "moderate":
        # Filter only if average rain is extremely low (e.g., < 1mm/day)
        if avg_daily_precip < 1.0:
            suitable = False
            log.info(
                f"[{crop_name}] Filtered Out (Water Check): Needs 'Moderate' water, but avg forecast rain ({avg_daily_precip:.2f}mm/day) seems insufficient (< 1.0)."
            )
    elif water_needs == "high":
        # Filter only if average rain is very low (e.g., < 2mm/day)
        if avg_daily_precip < 2.0:
            suitable = False
            log.info(
                f"[{crop_name}] Filtered Out (Water Check): Needs 'High' water, but avg forecast rain ({avg_daily_precip:.2f}mm/day) seems insufficient (< 2.0)."
            )
    elif water_needs == "very high":
        # Filter only if average rain is significantly low (e.g., < 3mm/day)
        if avg_daily_precip < 3.0:
            suitable = False
            log.info(
                f"[{crop_name}] Filtered Out (Water Check): Needs 'Very High' water, but avg forecast rain ({avg_daily_precip:.2f}mm/day) seems insufficient (< 3.0)."
            )

    if suitable:
        log.debug(f"[{crop_name}] Water check PASSED.")

    return suitable


# Function name reverted to original
def filter_agronomically_suitable_crops(
    potential_crops: List[str], input_data: InputParams
) -> List[str]:
    """
    Filters a list of potential crops based on basic agronomic suitability
    (soil type and water needs vs forecast check with lenient logic) using data
    from data_sources. Assumes uppercase crop names and soil types.

    XAI Component: Rule-Based System applying domain knowledge checks (Soil, Water).
    """
    suitable_crops = []
    # Updated logging to be more general
    log.info(
        f"--- Running Agronomic Filtering (Soil + Water) for {len(potential_crops)} potential crops ---"
    )
    log.info(
        f"Criteria: Soil Type='{input_data.soil_type}' (UPPERCASE), Lat/Lon='{input_data.latitude}/{input_data.longitude}', Forecast Length='{len(input_data.weather_forecast)} days'"
    )

    if not potential_crops:
        log.warning("No potential crops provided to filter.")
        return []

    # Ensure potential_crops are uppercase for lookup
    potential_crops_upper = [crop.upper() for crop in potential_crops]
    # Ensure input soil is uppercase for consistency in checks
    input_soil_upper = input_data.soil_type.upper()

    for crop_name in potential_crops_upper:
        log.debug(f"Checking agronomic suitability for: {crop_name}")
        # Fetch the detailed AgronomicInfo object for the crop (using uppercase name)
        agronomic_info: Optional[AgronomicInfo] = data_sources.get_crop_agronomic_data(
            crop_name
        )

        if not agronomic_info:
            log.warning(
                f"[{crop_name}] Filtered Out: No detailed agronomic data found."
            )
            continue

        # --- Check 1: Soil Suitability ---
        soil_ok = check_soil_suitability(agronomic_info, input_soil_upper)
        if not soil_ok:
            # Logging done within check_soil_suitability
            continue  # Move to the next crop

        # --- Check 2: Water Suitability (using the lenient logic internally) ---
        # Pass the full forecast list
        water_ok = check_water_suitability(agronomic_info, input_data.weather_forecast)
        if not water_ok:
            # Logging done within check_water_suitability
            continue  # Move to the next crop

        # --- If all checks passed ---
        # Updated logging
        log.info(f" + '{crop_name}': PASSED basic agronomic checks (Soil + Water).")
        suitable_crops.append(crop_name)  # Append the uppercase name

    # Updated logging
    log.info(f"--- Agronomic Filtering (Soil + Water) Complete ---")
    log.info(
        f"Found {len(suitable_crops)} agronomically suitable crops out of {len(potential_crops)} potential candidates."
    )
    log.debug(f"Suitable crops: {suitable_crops}")
    return suitable_crops
