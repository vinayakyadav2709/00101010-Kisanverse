# data_sources.py
import json
import os
import random
import logging
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime, timedelta

# Import config and models directly
import config
import models

log = logging.getLogger(__name__)

# --- Load ONLY Agronomic Database ---
AGRONOMIC_DATA: Dict[str, models.AgronomicInfo] = {}
# NEWS_DATA REMOVED - News will be fetched from ChromaDB by the pipeline/RAG components

def load_databases():
    """Loads ONLY the agronomic data from JSON file."""
    global AGRONOMIC_DATA # Removed NEWS_DATA
    log.info(f"Attempting to load agronomic data from: {config.AGRONOMIC_DB_PATH}")
    # --- Load Agronomic Data ---
    try:
        if os.path.exists(config.AGRONOMIC_DB_PATH):
            with open(config.AGRONOMIC_DB_PATH, 'r') as f:
                raw_data = json.load(f)
            validated_data = {}
            for crop, data in raw_data.items():
                try:
                    if 'crop_name' not in data: data['crop_name'] = crop
                    validated_data[crop] = models.AgronomicInfo(**data)
                except Exception as e:
                    log.warning(f"Skipping invalid agronomic data for '{crop}': {e}")
            AGRONOMIC_DATA = validated_data
            log.info(f"Loaded {len(AGRONOMIC_DATA)} crops from {config.AGRONOMIC_DB_PATH}")
        else:
            log.error(f"Agronomic DB not found: {config.AGRONOMIC_DB_PATH}. Run fetch script.")
            AGRONOMIC_DATA = {}
    except Exception as e:
        log.error(f"Failed load agronomic DB: {e}", exc_info=True)
        AGRONOMIC_DATA = {}

    # --- REMOVED News Data Loading Section ---

# Load only agronomic data when the module is imported
load_databases()

# --- Data Access Functions ---
def get_potential_crops(latitude: float, longitude: float) -> List[str]:
    """SIMULATED: Returns all known crops for now. Could add regional filtering later."""
    # TODO: Implement actual regional suitability filtering if needed
    if not AGRONOMIC_DATA:
        log.warning("Agronomic data empty. Cannot determine potential crops.")
        return []
    # For now, return all crops keys as potential candidates
    return list(AGRONOMIC_DATA.keys())

def get_crop_agronomic_data(crop_name: str) -> Optional[models.AgronomicInfo]:
    """Gets AgronomicInfo object for a crop."""
    return AGRONOMIC_DATA.get(crop_name)

def get_yield_estimate(crop_name: str, soil_type: str) -> str:
    """Gets yield estimate string, potentially adjusted for soil."""
    agronomic_info = AGRONOMIC_DATA.get(crop_name)
    if agronomic_info and agronomic_info.typical_yield_range_q_acre:
        low, high = agronomic_info.typical_yield_range_q_acre
        # Basic adjustment: Reduce yield slightly if soil isn't listed as suitable
        if soil_type not in agronomic_info.suitable_soils:
            low = max(1.0, low * 0.8) # Reduce lower bound by 20%, ensure it's at least 1.0
            high *= 0.9             # Reduce upper bound by 10%
            log.debug(f"[{crop_name}] Soil '{soil_type}' not listed as suitable. Adjusted yield: {low:.1f}-{high:.1f} q/acre")
        return f"{low:.1f}-{high:.1f} q/acre" # Use quintals/acre consistently
    return "N/A"


def get_price_forecast(crop_name: str) -> Dict:
    """SIMULATED: Generates a plausible price forecast trend and volatility."""
    log.info(f"SIM: Generating price forecast for '{crop_name}'")
    trends = ["Stable", "Slightly Rising", "Rising", "Falling", "Volatile"]
    volatility = ["Low", "Medium", "High"]

    # Example logic based on crop type (can be made more sophisticated)
    if "Maize" in crop_name or "Soybean" in crop_name:
        trend = random.choice(["Rising", "Stable", "Rising"]) # Higher demand/exports?
    elif "Cotton" in crop_name:
        trend = random.choice(["Stable", "Volatile", "Falling"]) # Often volatile
    elif "Paddy" in crop_name or "Wheat" in crop_name:
        trend = random.choice(["Stable", "Slightly Rising"]) # Often MSP influenced
    else:
        trend = random.choice(trends) # Default random trend

    # Adjust volatility based on trend
    vol = random.choice(volatility)
    if trend == "Volatile":
        vol = "High"
    elif trend == "Stable":
        vol = "Low"

    log.debug(f"[{crop_name}] Simulated forecast: Trend={trend}, Volatility={vol}")
    return {"trend": trend, "volatility": vol}


def get_input_cost_category(crop_name: str) -> str:
    """Gets the input cost category from agronomic data."""
    agronomic_info = AGRONOMIC_DATA.get(crop_name)
    return agronomic_info.input_cost_cat if agronomic_info else "Unknown"

def get_relevant_events(latitude: float, longitude: float, crops: List[str]) -> List[Dict]:
    """SIMULATED: Checks for critical alerts (e.g., pest outbreaks)."""
    log.info(f"SIM: Checking for critical events impacting {len(crops)} crops...")
    events = []
    region = config.get_region_code(latitude, longitude)

    # Example Simulated Events (make these more dynamic or data-driven if possible)
    if random.random() < 0.05: # Small chance of a generic high severity event
         events.append({
            "type": "Weather Alert",
            "crop_affected": [], # Affects all crops in region
            "severity": "High",
            "region_codes": [region],
            "summary": f"Simulated HIGH severity weather warning (e.g., cyclone) in region {region}."
         })

    for crop in crops:
        agronomic_info = AGRONOMIC_DATA.get(crop)
        if agronomic_info:
            # Example: Higher chance of specific pest alert if it's common
            if "bollworm" in str(agronomic_info.common_pests).lower() and random.random() < 0.1:
                events.append({
                    "type": "Pest Alert",
                    "crop_affected": [crop],
                    "severity": random.choice(["Medium", "High"]),
                    "region_codes": [region],
                    "summary": f"Simulated alert for Bollworm in {crop}."
                })
            if "armyworm" in str(agronomic_info.common_pests).lower() and random.random() < 0.15:
                events.append({
                    "type": "Pest Alert",
                    "crop_affected": [crop, "Maize (Kharif)"], # Can affect multiple crops
                    "severity": "Medium",
                    "region_codes": [region],
                    "summary": f"Simulated MEDIUM alert for Armyworm."
                 })
            # Example: Disease alert simulation
            if "blight" in str(agronomic_info.common_pests).lower() and random.random() < 0.08:
                 events.append({
                    "type": "Disease Alert",
                    "crop_affected": [crop],
                    "severity": "Medium",
                    "region_codes": [region],
                    "summary": f"Simulated alert for potential Blight issue in {crop}."
                 })

    log.info(f"Found {len(events)} simulated relevant events for the region/crops.")
    return events


# --- fetch_relevant_news REMOVED ---
# News retrieval will happen in the pipeline using ChromaDB RAG


def recommend_pesticides(crop_name: str, risks: List[str]) -> List[models.PesticideSuggestion]:
    """Recommends pesticides based on common pests and identified risks/alerts."""
    log.info(f"Recommending pesticides for '{crop_name}' based on common pests and risks: {risks}")
    recommendations = {} # Use dict to avoid duplicate chemicals, update targets/timing
    agronomic_info = AGRONOMIC_DATA.get(crop_name)
    common_pests = [p.lower() for p in agronomic_info.common_pests] if agronomic_info else []

    # Simplified Pest->Chemical mapping (Expand this significantly for real use)
    # Keys are chemical names, values are dicts of {pest_keyword: timing_stage}
    pest_map = {
        "Thiamethoxam": {"girdle beetle": "Vegetative", "jassids": "Vegetative", "whitefly": "Early", "aphids": "Early", "stem borer": "Early"},
        "Profenofos": {"girdle beetle": "Podding", "bollworm": "Boll Stage", "semilooper": "Vegetative"},
        "Imidacloprid": {"whitefly": "Seedling/Early", "aphids": "Seedling/Early", "shoot fly": "Seed Treatment", "termite": "Seed Treatment/Soil"},
        "Acetamiprid": {"whitefly": "Vegetative", "aphids": "Vegetative", "thrips": "Flowering"},
        "Emamectin Benzoate": {"armyworm": "Vegetative/Reproductive", "pod borer": "Flowering/Podding", "fruit borer": "Fruiting", "bollworm": "Boll Stage", "leaf folder": "Vegetative"},
        "Spinetoram": {"armyworm": "Vegetative", "thrips": "Flowering", "leaf miner": "Vegetative"},
        "Cartap Hydrochloride": {"stem borer": "Vegetative", "leaf folder": "Vegetative"},
        "Fipronil": {"stem borer": "Vegetative", "thrips": "Flowering", "termite": "Soil Application", "white grub": "Soil Application"},
        "Chlorantraniliprole (Coragen)": {"bollworm": "Flowering/Boll", "pod borer": "Flowering/Podding", "stem borer": "Early", "leaf folder": "Vegetative", "armyworm": "Vegetative"},
        "Lambda-cyhalothrin": {"bollworm": "Flowering/Boll", "jassids": "Vegetative", "aphids": "Vegetative", "semilooper": "Vegetative"},
        "Dimethoate": {"aphids": "Vegetative", "jassids": "Early", "thrips": "Flowering", "mites": "Any Stage"},
        "Acephate": {"aphids": "Vegetative", "thrips": "Flowering", "jassids": "Vegetative"},
        "Flubendiamide": {"leaf folder": "Vegetative", "pod borer": "Podding", "bollworm": "Boll Stage"},
        "Indoxacarb": {"pod borer": "Flowering/Podding", "semilooper": "Vegetative", "fruit borer": "Fruiting"},
        "Malathion": {"fruit fly": "Fruiting (Bait)", "general pest": "General Spray"},
        "Spinosad": {"thrips": "Flowering", "leaf miner": "Vegetative", "pod borer": "Flowering"},
        "Buprofezin": {"mealybug": "Vegetative", "whitefly": "Nymph Stage", "jassids": "Nymph Stage"},
        "Diafenthiuron": {"jassids": "Vegetative", "whitefly": "Adult Stage", "mites": "Any Stage"},
        "Chlorpyrifos": {"termite": "Soil Application (Pre/Post-sowing)", "cutworm": "Soil Drench", "white grub": "Soil Drench", "stem borer": "Basal Spray"},
        # Fungicides (Add if needed, map to diseases)
        "Mancozeb": {"blight": "Preventative/Early", "rust": "Preventative/Early"},
        "Carbendazim": {"wilt": "Seed Treatment/Drench", "powdery mildew": "Foliar Spray"},
        "Propiconazole": {"rust": "Foliar Spray", "leaf spot": "Foliar Spray"}
    }

    def add_recommendation(chem, pest, stage, reason):
        """Adds or updates a pesticide recommendation."""
        key = chem # Use chemical name as key
        target_text = f"{pest} ({reason})" # Include reason (common/alert)
        if key not in recommendations:
            recommendations[key] = models.PesticideSuggestion(
                chemical_name=chem,
                target_pest=target_text,
                timing_stage=stage
            )
        else:
            # Append pest/reason if not already listed for this chemical
            if pest not in recommendations[key].target_pest:
                recommendations[key].target_pest += f"; {target_text}"
            # Could potentially update timing if alert suggests a different stage
            # recommendations[key].timing_stage = stage # Uncomment to override timing

    # 1. Process common pests for the crop
    for pest_key in common_pests:
        # Find chemicals that target this common pest
        for chem, targets in pest_map.items():
            if pest_key in targets:
                add_recommendation(chem, pest_key, targets[pest_key], "common")

    # 2. Process identified risks/alerts
    for risk_desc in risks:
        risk_lower = risk_desc.lower()
        # Check if risk description mentions a known pest keyword
        for chem, targets in pest_map.items():
            for pest_keyword, stage in targets.items():
                if pest_keyword in risk_lower:
                    # If risk mentions 'bollworm', add chemicals targeting 'bollworm'
                    add_recommendation(chem, pest_keyword, stage, "alert") # Mark as triggered by alert

    # Finalize and limit results
    final_list = sorted(list(recommendations.values()), key=lambda x: x.chemical_name)
    log.info(f"Suggested {len(final_list)} potential pesticide options for '{crop_name}'. Limiting to top 5.")
    # Log first few suggestions for debugging
    log.debug(f"Top suggestions for '{crop_name}': {[(p.chemical_name, p.target_pest, p.timing_stage) for p in final_list[:5]]}")
    return final_list[:5] # Return only top 5 suggestions

def get_subsidy_details_for_crop(crop_name: str, input_subsidies: List[Dict], region_code: str) -> List[Dict]:
    """Filters input subsidies list for relevance to the specific crop and region."""
    log.info(f"Filtering {len(input_subsidies)} provided subsidies for '{crop_name}' in region '{region_code}'")
    relevant = []
    crop_name_lower = crop_name.lower().split(" ")[0] # Use first word of crop name for matching
    # Generic terms that might indicate relevance even if crop name isn't exact match
    generic_terms = ["seed", "fertilizer", "input", "cultivation", "irrigation", "organic", "pulse", "oilseed", "cereal", "kharif", "rabi", "farmer", "agriculture", "horticulture"]

    for sub in input_subsidies:
        # Check Region Applicability
        applies_to_region = True # Assume applies if locations field is missing or empty
        if sub.get("locations"):
            applies_to_region = region_code in sub.get("locations", [])
        if not applies_to_region:
            # log.debug(f"Subsidy '{sub.get('program', 'N/A')}' skipped: Not applicable to region '{region_code}' (Applies to: {sub.get('locations')})")
            continue

        # Check Crop Relevance (Search in program name, description, benefits)
        program_lower = sub.get("program", "").lower()
        desc_lower = sub.get("description", "").lower()
        benefits_lower = sub.get("benefits", "").lower()
        text_to_search = program_lower + " " + desc_lower + " " + benefits_lower

        is_relevant_to_crop = False
        if crop_name_lower in text_to_search:
            is_relevant_to_crop = True
        else:
            # Check for generic terms only if specific crop name not found
            if any(term in text_to_search for term in generic_terms):
                 is_relevant_to_crop = True # Could be general input subsidy

        if is_relevant_to_crop:
            log.debug(f"Subsidy '{sub.get('program', 'N/A')}' identified as relevant for '{crop_name}'.")
            # Format output for consistency
            relevant.append({
                "program": sub.get("program", "N/A"),
                "provider": sub.get("provider", "N/A"),
                "benefit_summary": sub.get("benefits", "N/A"),
                "estimated_value_inr": sub.get("value_estimate_inr") # Pass through estimate if available
            })

    log.info(f"Found {len(relevant)} relevant subsidies for '{crop_name}' based on provided list.")
    return relevant


# --- Plotting Data Generation (Return DATE OBJECTS for price chart) ---
def generate_price_chart_data(crop_name: str, forecast: Dict) -> List[Dict]:
    """Generates simulated price chart data for future dates, returning date OBJECTS."""
    log.info(f"SIM: Generating price plot data for {crop_name}")
    agronomic_info = AGRONOMIC_DATA.get(crop_name)
    cost_cat = agronomic_info.input_cost_cat if agronomic_info else "Medium"

    # Base price simulation (could be more sophisticated, e.g., using historical data)
    base_mult = {"Low": 1.5, "Low-Medium": 1.8, "Medium": 2.2, "Medium-High": 2.5, "High": 3.0}.get(cost_cat, 2.0)
    # Simulate a base price per quintal (adjust range as needed)
    base = int(random.uniform(1500, 4000) * base_mult) # Wider range for base price simulation
    data = []

    # Simulate for future dates (e.g., ~4, 5, 6, 7 months from now)
    dates_to_plot = [(datetime.now().date() + timedelta(days=d)) for d in [120, 150, 180, 210]]
    trend_factor = 1.0
    if "Rising" in forecast.get('trend', ''): trend_factor = 1.03 # Simulate 3% increase per period
    if "Falling" in forecast.get('trend', ''): trend_factor = 0.97 # Simulate 3% decrease per period

    current_min = base * 0.9
    current_max = base * 1.1

    for date_obj in dates_to_plot:
        # Apply trend
        current_min = int(current_min * trend_factor)
        current_max = int(current_max * trend_factor)

        # Apply volatility spread
        volatility = forecast.get("volatility", "Medium")
        spread = (current_max - current_min) * {"Low": 0.1, "Medium": 0.2, "High": 0.35}.get(volatility, 0.2)
        plot_min = max(0, int(current_min - spread / 2)) # Ensure min price is not negative
        plot_max = int(current_max + spread / 2)

        # Ensure min is less than or equal to max
        if plot_min > plot_max:
            plot_min, plot_max = plot_max, plot_min # Swap if needed

        data.append({
            "date": date_obj, # Append date object directly
            "predicted_price_min": plot_min,
            "predicted_price_max": plot_max
        })

    log.debug(f"Generated price chart data points for {crop_name}: {len(data)}")
    return data


def generate_water_chart_data(crop_name: str) -> List[Dict]:
    """Generates simple water need levels by typical growth stage."""
    log.info(f"SIM: Generating water need plot data for {crop_name}")
    agronomic_info = AGRONOMIC_DATA.get(crop_name)
    # Map general water needs to a base level (1-5 scale)
    base_need_map = {"Low": 1, "Moderate": 2, "High": 4, "Very High": 5}
    base_need = base_need_map.get(agronomic_info.water_needs if agronomic_info else "Moderate", 2)

    # Define typical stages and relative need multipliers
    stages = [
        ("Germination/Establishment", base_need * 0.8),
        ("Vegetative Growth", base_need * 1.2),
        ("Flowering/Reproductive", base_need * 1.5), # Peak water demand often here
        ("Maturity/Harvest", base_need * 0.5)
    ]

    data = []
    for stage, level_multiplier in stages:
        # Calculate relative need level, ensuring it's within 1-5 range
        relative_level = max(1, min(5, round(level_multiplier)))
        data.append({"growth_stage": stage, "relative_need_level": relative_level})

    log.debug(f"Generated water chart data for {crop_name}: {data}")
    return data

def generate_fertilizer_chart_data(crop_name: str) -> List[Dict]:
    """Generates a typical fertilizer schedule based on needs."""
    log.info(f"SIM: Generating fertilizer schedule plot data for {crop_name}")
    agronomic_info = AGRONOMIC_DATA.get(crop_name)
    fert_needs = agronomic_info.fertilizer_needs if agronomic_info else "Standard NPK"
    water_needs = agronomic_info.water_needs if agronomic_info else "Moderate"

    # Base schedule
    schedule = [{"stage": "Basal", "timing": "At/Before Sowing", "nutrients": f"Foundation Dose ({fert_needs})"}]

    # Add top dressing based on needs (example logic)
    if "High N" in fert_needs or "Very High N" in fert_needs or "High NPK" in fert_needs:
        schedule.append({"stage": "Top Dress 1", "timing": "~25-40 DAS", "nutrients": "Nitrogen Focus (e.g., Urea)"}) # Days After Sowing

    # Add second top dressing for demanding crops or specific needs
    if "Very High NPK" in fert_needs or "High" in water_needs or "Micronutrients" in fert_needs or "Long duration" in (agronomic_info.general_notes or ""):
         schedule.append({"stage": "Top Dress 2 / Foliar", "timing": "~50-75 DAS", "nutrients": "NPK Balance / Potassium / Micronutrients as needed"})

    # Add stage for specific nutrient timings if mentioned
    if "Sulphur" in fert_needs:
         # Find if already added, otherwise add specific note or stage
         found = False
         for item in schedule:
             if "Sulphur" in item["nutrients"]: found = True; break
         if not found: schedule.append({"stage": "Nutrient Specific", "timing": "Basal/Early", "nutrients": "Sulphur Application"})

    if "Zinc" in fert_needs:
        found = False
        for item in schedule:
             if "Zinc" in item["nutrients"]: found = True; break
        if not found: schedule.append({"stage": "Nutrient Specific", "timing": "Basal/Foliar", "nutrients": "Zinc Application"})


    log.debug(f"Generated fertilizer chart data for {crop_name}: {schedule}")
    return schedule