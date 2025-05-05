# fetch_agronomic_data.py
import json
import os
import random
import logging
from typing import Dict, List, Tuple, Optional

# Import config and models directly (flat structure)
import models.llm.config as config  # Contains DB paths
import models.llm.models as models  # Ensure models.py with AgronomicInfo is present

log = logging.getLogger(__name__)

# --- Plausible Agronomic Data ---
# Added more detail, more crops, and ensured consistency
PREDEFINED_AGRONOMIC_DATA = {
    "Paddy (Rice - Kharif)": {
        "suitable_soils": ["Clayey", "Clay Loam", "Alluvial", "Silty Clay"],
        "water_needs": "Very High",
        "input_cost_cat": "Medium",
        "fertilizer_needs": "High N, Medium P, Medium K, Zinc essential",
        "common_pests": [
            "stem borer",
            "leaf folder",
            "brown planthopper (BPH)",
            "gall midge",
            "bacterial blight",
        ],
        "typical_yield_range_q_acre": (18.0, 32.0),
        "general_notes": "Requires puddling and standing water for optimal growth. Sensitive to salinity.",
    },
    "Wheat (Rabi)": {
        "suitable_soils": ["Loamy", "Clay Loam", "Alluvial", "Sandy Loam (irrigated)"],
        "water_needs": "Moderate",
        "input_cost_cat": "Medium",
        "fertilizer_needs": "High N, Medium P, Low-Medium K, Sulphur often needed",
        "common_pests": [
            "aphids",
            "termite",
            "rust (yellow, brown, black)",
            "karnal bunt",
        ],
        "typical_yield_range_q_acre": (16.0, 28.0),
        "general_notes": "Cool weather needed for germination and tillering. Heat stress sensitive during grain filling.",
    },
    "Maize (Kharif)": {
        "suitable_soils": ["Loamy", "Sandy Loam", "Alluvial", "Black", "Red Loam"],
        "water_needs": "High",
        "input_cost_cat": "Medium-High",
        "fertilizer_needs": "Very High N, High P, Medium K, Zinc",
        "common_pests": ["fall armyworm", "stem borer", "shoot fly", "cob borer"],
        "typical_yield_range_q_acre": (20.0, 35.0),
        "general_notes": "Sensitive to waterlogging, especially early. Critical water need at tasseling/silking.",
    },
    "Maize (Rabi)": {
        "suitable_soils": ["Loamy", "Sandy Loam", "Alluvial", "Black", "Clay Loam"],
        "water_needs": "Moderate",
        "input_cost_cat": "Medium",
        "fertilizer_needs": "High N, High P, Medium K, Zinc",
        "common_pests": ["stem borer", "aphids", "pink borer"],
        "typical_yield_range_q_acre": (25.0, 45.0),
        "general_notes": "Requires assured irrigation. Generally higher yield potential than Kharif.",
    },
    "Soybean": {
        "suitable_soils": ["Black", "Loamy", "Clay Loam", "Alluvial (well-drained)"],
        "water_needs": "Moderate",
        "input_cost_cat": "Low-Medium",
        "fertilizer_needs": "Rhizobium Inoculant essential, Low N starter, High P, Medium K, Sulphur",
        "common_pests": [
            "girdle beetle",
            "whitefly",
            "pod borer complex",
            "semilooper",
            "yellow mosaic virus",
        ],
        "typical_yield_range_q_acre": (8.0, 16.0),
        "general_notes": "Nitrogen fixing legume. Sensitive to waterlogging. Needs good drainage.",
    },
    "Cotton": {
        "suitable_soils": [
            "Black (Deep)",
            "Alluvial (well-drained)",
            "Red Sandy Loam (irrigated)",
        ],
        "water_needs": "High",
        "input_cost_cat": "High",
        "fertilizer_needs": "Very High NPK, Magnesium, Boron, Zinc",
        "common_pests": [
            "pink bollworm",
            "american bollworm",
            "jassids",
            "whitefly",
            "thrips",
            "leaf curl virus",
        ],
        "typical_yield_range_q_acre": (4.0, 12.0),
        "general_notes": "Long duration crop. Requires intensive pest management (IPM). Sensitive to water stress at flowering/boll development.",
    },
    "Sugarcane": {
        "suitable_soils": ["Loamy", "Clay Loam", "Black", "Alluvial"],
        "water_needs": "Very High",
        "input_cost_cat": "High",
        "fertilizer_needs": "Very High NPK, High Organic Matter, Sulphur, Iron",
        "common_pests": [
            "early shoot borer",
            "top borer",
            "internode borer",
            "termites",
            "whitefly",
            "red rot disease",
        ],
        "typical_yield_range_q_acre": (250.0, 500.0),
        "general_notes": "Long duration (10-18 months). Ratooning common. High water and nutrient demanding.",
    },
    "Groundnut (Kharif)": {
        "suitable_soils": ["Sandy Loam", "Loamy Sand", "Red Loam"],
        "water_needs": "Moderate",
        "input_cost_cat": "Medium",
        "fertilizer_needs": "Rhizobium Inoculant, Low N starter, High P, Medium K, Gypsum/Calcium essential for pod development",
        "common_pests": [
            "leaf miner",
            "thrips",
            "white grub",
            "collar rot",
            "tikka disease",
        ],
        "typical_yield_range_q_acre": (8.0, 18.0),
        "general_notes": "Requires well-drained, light soils. Calcium application at pegging stage is critical.",
    },
    "Mustard (Rapeseed-Mustard - Rabi)": {
        "suitable_soils": ["Sandy Loam", "Loamy", "Alluvial"],
        "water_needs": "Low",
        "input_cost_cat": "Low-Medium",
        "fertilizer_needs": "Medium N, Low P, Low K, Sulphur essential",
        "common_pests": ["aphids", "sawfly", "alternaria blight", "white rust"],
        "typical_yield_range_q_acre": (6.0, 14.0),
        "general_notes": "Major Rabi oilseed. Tolerant to frost to some extent. Sulphur crucial for oil content.",
    },
    "Chickpea (Gram - Rabi)": {
        "suitable_soils": ["Loamy", "Sandy Loam", "Clay Loam", "Black (light-medium)"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Rhizobium Inoculant, Low N starter, Medium P, Low K",
        "common_pests": [
            "pod borer (Helicoverpa)",
            "cutworm",
            "termite",
            "wilt disease",
            "ascochyta blight",
        ],
        "typical_yield_range_q_acre": (6.0, 12.0),
        "general_notes": "Major pulse crop. Often grown on residual moisture. Sensitive to waterlogging and excessive cold/frost.",
    },
    "Pigeonpea (Tur/Arhar - Kharif)": {
        "suitable_soils": ["Loamy", "Sandy Loam", "Red Loam", "Black (light)"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Rhizobium Inoculant, Low N starter, Medium P, Low K",
        "common_pests": [
            "pod borer complex",
            "pod fly",
            "maruca",
            "wilt disease",
            "sterility mosaic disease",
        ],
        "typical_yield_range_q_acre": (4.0, 9.0),
        "general_notes": "Deep-rooted, drought tolerant pulse. Often intercropped. Long duration variants exist.",
    },
    "Millet (Bajra/Pearl Millet - Kharif)": {
        "suitable_soils": ["Sandy", "Sandy Loam", "Loamy Sand", "Light Black"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Low NPK, responds well to FYM",
        "common_pests": ["shoot fly", "stem borer", "downy mildew", "ergot", "smut"],
        "typical_yield_range_q_acre": (6.0, 14.0),
        "general_notes": "Highly drought tolerant cereal. Suitable for arid and semi-arid regions.",
    },
    "Sorghum (Jowar - Kharif/Rabi)": {
        "suitable_soils": ["Loamy", "Clay Loam", "Black", "Red Loam"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Medium N, Low P, Low K",
        "common_pests": ["shoot fly", "stem borer", "midge", "head smut", "grain mold"],
        "typical_yield_range_q_acre": (8.0, 22.0),
        "general_notes": "Dual purpose (grain/fodder). Tolerant to drought and moderate salinity.",
    },
    "Potato (Rabi/Hills)": {
        "suitable_soils": [
            "Sandy Loam",
            "Loamy",
            "Alluvial",
            "Silt Loam (well-drained)",
        ],
        "water_needs": "High",
        "input_cost_cat": "High",
        "fertilizer_needs": "Very High NPK, High Organic Matter, Calcium, Magnesium",
        "common_pests": [
            "aphids (virus vector)",
            "tuber moth",
            "cutworm",
            "late blight",
            "early blight",
            "black scurf",
        ],
        "typical_yield_range_q_acre": (80.0, 160.0),
        "general_notes": "Requires quality seed tubers (disease-free). Sensitive to frost and high temperatures during tuberization.",
    },
    "Onion (Rabi/Kharif)": {
        "suitable_soils": [
            "Sandy Loam",
            "Loamy",
            "Silt Loam",
            "Light Black (well-drained)",
        ],
        "water_needs": "Moderate",
        "input_cost_cat": "Medium-High",
        "fertilizer_needs": "Medium N, High P, High K, Sulphur essential for pungency",
        "common_pests": ["thrips", "maggots", "purple blotch", "basal rot"],
        "typical_yield_range_q_acre": (60.0, 130.0),
        "general_notes": "Sensitive to day length for bulbing. Requires good drainage.",
    },
    "Tomato (Kharif/Rabi/Summer)": {
        "suitable_soils": ["Sandy Loam", "Loamy", "Clay Loam", "Black (well-drained)"],
        "water_needs": "High",
        "input_cost_cat": "High",
        "fertilizer_needs": "High NPK, Calcium important, Boron",
        "common_pests": [
            "fruit borer (Helicoverpa)",
            "whitefly (virus vector)",
            "leaf miner",
            "mites",
            "leaf curl virus",
            "bacterial wilt",
            "early blight",
            "late blight",
        ],
        "typical_yield_range_q_acre": (100.0, 250.0),
        "general_notes": "Requires staking for indeterminate varieties. Sensitive to water stress and temperature extremes.",
    },
    "Sunflower (Kharif/Rabi/Summer)": {
        "suitable_soils": ["Sandy Loam", "Loamy", "Black", "Alluvial"],
        "water_needs": "Moderate",
        "input_cost_cat": "Medium",
        "fertilizer_needs": "Medium NPK, Boron essential for seed set",
        "common_pests": [
            "head borer (capitulum borer)",
            "jassids",
            "whitefly",
            "alternaria blight",
            "powdery mildew",
        ],
        "typical_yield_range_q_acre": (6.0, 11.0),
        "general_notes": "Short duration oilseed. Relatively photo-insensitive. Needs sufficient moisture during flowering.",
    },
    "Lentil (Masoor - Rabi)": {
        "suitable_soils": ["Loamy", "Clay Loam", "Alluvial", "Silty Loam"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Rhizobium Inoculant, Low N starter, Medium P, Low K",
        "common_pests": ["aphids", "pod borer", "wilt", "rust"],
        "typical_yield_range_q_acre": (4.0, 9.0),
        "general_notes": "Often grown on residual moisture in rice fallows. Tolerant to cold.",
    },
    "Barley (Rabi)": {
        "suitable_soils": ["Sandy Loam", "Loamy", "Saline/Alkaline soils (tolerant)"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Low-Medium NPK",
        "common_pests": [
            "aphids",
            "termite",
            "covered smut",
            "loose smut",
            "stripe disease",
        ],
        "typical_yield_range_q_acre": (10.0, 20.0),
        "general_notes": "Hardy cereal. More tolerant to salinity, alkalinity, and drought than wheat.",
    },
    "Sesame (Til - Kharif/Summer)": {
        "suitable_soils": ["Sandy Loam", "Loamy", "Light Red"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Low NPK",
        "common_pests": [
            "leaf roller",
            "capsule borer",
            "gall fly",
            "phyllody disease",
        ],
        "typical_yield_range_q_acre": (2.0, 5.0),
        "general_notes": "Ancient oilseed. Requires warm climate. Sensitive to waterlogging.",
    },
    "Moong Bean (Green Gram - Kharif/Summer/Spring)": {
        "suitable_soils": ["Sandy Loam", "Loamy", "Alluvial"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Rhizobium Inoculant, Low N starter, Medium P",
        "common_pests": [
            "pod borer",
            "jassids",
            "whitefly",
            "yellow mosaic virus (YMV)",
        ],
        "typical_yield_range_q_acre": (3.0, 7.0),
        "general_notes": "Short duration pulse (60-70 days). Fits well in crop rotations. YMV is a major constraint.",
    },
    "Urad Bean (Black Gram - Kharif)": {
        "suitable_soils": ["Clay Loam", "Loamy", "Black", "Alluvial"],
        "water_needs": "Low",
        "input_cost_cat": "Low",
        "fertilizer_needs": "Rhizobium Inoculant, Low N starter, Medium P",
        "common_pests": [
            "pod borer",
            "aphids",
            "jassids",
            "whitefly",
            "yellow mosaic virus (YMV)",
        ],
        "typical_yield_range_q_acre": (3.0, 6.0),
        "general_notes": "Prefers heavier soils than Moong bean. Important pulse crop.",
    },
    "Jute (Kharif)": {
        "suitable_soils": ["Alluvial", "Loamy", "Clay Loam"],
        "water_needs": "High",
        "input_cost_cat": "Medium",
        "fertilizer_needs": "High N, Low P, Medium K",
        "common_pests": ["semilooper", "stem weevil", "yellow mite", "stem rot"],
        "typical_yield_range_q_acre": (8.0, 16.0),
        "general_notes": "Major fibre crop. Requires high humidity and rainfall. Needs retting process post-harvest.",
    },
    "Castor (Kharif/Perennial)": {
        "suitable_soils": ["Sandy Loam", "Red Loam", "Light Black"],
        "water_needs": "Low",
        "input_cost_cat": "Low-Medium",
        "fertilizer_needs": "Medium NPK",
        "common_pests": ["capsule borer", "semilooper", "whitefly", "wilt"],
        "typical_yield_range_q_acre": (5.0, 12.0),
        "general_notes": "Non-edible oilseed. Highly drought tolerant. Suitable for marginal lands.",
    },
    "Chilli (Kharif/Rabi/Summer)": {
        "suitable_soils": ["Loamy", "Black (well-drained)", "Sandy Loam", "Laterite"],
        "water_needs": "Moderate",
        "input_cost_cat": "Medium-High",
        "fertilizer_needs": "Medium N, High P, High K, Calcium, Boron",
        "common_pests": [
            "thrips",
            "mites",
            "aphids",
            "fruit borer",
            "leaf curl virus",
            "anthracnose",
            "powdery mildew",
        ],
        "typical_yield_range_q_acre": (6.0, 18.0),
        "general_notes": "Requires warm, humid climate initially, dry during fruit maturity. Sensitive to water stress and frost.",
    },
}


def generate_and_validate_data() -> Dict[str, Dict]:
    """Validates the predefined data against the Pydantic model and returns validated data."""
    db = {}
    log.info(
        f"Validating predefined agronomic data for {len(PREDEFINED_AGRONOMIC_DATA)} crops..."
    )
    valid_count = 0
    invalid_count = 0
    for crop_name, data in PREDEFINED_AGRONOMIC_DATA.items():
        try:
            # Ensure crop_name field is present in data dict before validation
            if "crop_name" not in data:
                data["crop_name"] = crop_name
            # Validate data against the Pydantic model
            crop_info = models.AgronomicInfo(**data)
            # Store the validated data as a dictionary
            db[crop_name] = crop_info.dict()  # Use .dict() for JSON serializable output
            valid_count += 1
        except Exception as e:
            log.error(f"Validation failed for predefined crop '{crop_name}': {e}")
            invalid_count += 1

    if invalid_count > 0:
        log.warning(f"Validation finished with {invalid_count} errors.")
    else:
        log.info(f"Validation finished successfully for all {valid_count} crops.")
    return db


def save_data(data: Dict, path: str):
    """Saves the validated data dictionary to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write data to JSON file
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        log.info(
            f"Successfully saved validated agronomic data ({len(data)} crops) to {path}"
        )
    except Exception as e:
        log.error(f"Failed to save data to {path}: {e}")


if __name__ == "__main__":
    log.info("--- Generating Validated Agronomic Database from Predefined Data ---")
    # Ensure data directory exists
    if not os.path.exists(config.DATA_DIR):
        log.info(f"Creating data directory: {config.DATA_DIR}")
        os.makedirs(config.DATA_DIR)

    # Generate and validate data
    agronomic_data = generate_and_validate_data()

    # Save if data is valid
    if agronomic_data:
        save_data(agronomic_data, config.AGRONOMIC_DB_PATH)
    else:
        log.error(
            "No valid agronomic data was generated. Database file not saved/updated."
        )

    log.info("--- Finished Agronomic Data Generation ---")
