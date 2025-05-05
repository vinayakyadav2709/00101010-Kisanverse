# main.py
import json
import logging
import os
import sys
import traceback
from pydantic import ValidationError

# Import directly (flat structure)
import models # Contains InputParams
import pipeline
import config
import data_sources # To explicitly load DBs before pipeline
import fetch_news_data # To initialize vector DB connection if needed
# --- MODIFIED: Add import for explanation module (still needed to call its functions) ---
import explanation
# --- END MODIFIED ---

log = logging.getLogger(__name__)

INPUT_FILE = "ss2.json" # Default input file name
OUTPUT_FILE = "recommendation_output3.json"

def adapt_input_data(raw_data: dict) -> dict:
    """Adapts the raw input JSON structure (ss.json) to match the InputParams model."""
    log.info("Adapting ss.json input data structure...")
    adapted = {}
    try:
        # Direct mapping for simple fields
        adapted["soil_type"] = raw_data["soil_type"]
        adapted["latitude"] = raw_data["latitude"]
        adapted["longitude"] = raw_data["longitude"]
        adapted["planting_date_estimate"] = raw_data["start_date"] # Map start_date
        adapted["land_size_acres"] = float(raw_data["land_size"])

        # Adapt subsidies list - Extract only relevant fields, ignore others
        adapted["available_subsidies"] = []
        for sub_in in raw_data.get("subsidies", []):
            adapted_sub = {
                "type": sub_in.get("type", "Unknown"),
                "provider": sub_in.get("provider", "Unknown"),
                "program": sub_in.get("program"),
                "description": sub_in.get("description"),
                "eligibility": sub_in.get("eligibility"),
                "benefits": sub_in.get("benefits"),
                "locations": sub_in.get("locations", []),
                "value_estimate_inr": None
            }
            if adapted_sub["program"] and adapted_sub["benefits"]:
                adapted["available_subsidies"].append(adapted_sub)
            else:
                log.warning(f"Skipping subsidy due to missing program/benefits: {sub_in.get('$id', 'Unknown ID')}")


        # Adapt weather forecast list
        adapted["weather_forecast"] = []
        for wf_in in raw_data.get("weather_predictions", []):
            adapted["weather_forecast"].append({
                "date": wf_in["date"],
                "temp_max": wf_in["temperature_2m_max"],
                "temp_min": wf_in["temperature_2m_min"],
                "precip": wf_in["precipitation_sum"],
                "wind_max": wf_in.get("wind_speed_10m_max"),
                "radiation": wf_in.get("shortwave_radiation_sum")
            })

        log.debug(f"Finished adapting input data. Keys: {adapted.keys()}")
        log.debug(f"Adapted {len(adapted['available_subsidies'])} subsidies.")
        return adapted

    except KeyError as e:
        log.error(f"Missing expected key in input file '{INPUT_FILE}': {e}")
        raise ValueError(f"Input JSON is missing expected key: {e}")
    except ValueError as e:
         log.error(f"Type conversion error during input adaptation (e.g., land_size): {e}")
         raise ValueError(f"Invalid data type in input JSON: {e}")
    except Exception as e:
        log.error(f"Unexpected error adapting input data: {e}", exc_info=True)
        raise ValueError(f"Error adapting input data: {e}")


if __name__ == "__main__":
    log.info("=============================================")
    log.info("=== Starting Crop Recommendation Pipeline ===")
    log.info("=============================================")

    # --- Pre-flight Checks ---
    if not os.path.exists(config.AGRONOMIC_DB_PATH):
        log.error(f"Agronomic database file missing: {config.AGRONOMIC_DB_PATH}")
        print(f"\nERROR: Agronomic data file not found at '{config.AGRONOMIC_DB_PATH}'.")
        print("Please run: python fetch_agronomic_data.py")
        sys.exit(1)
    if not os.path.exists(config.CHROMA_DB_PATH):
         log.warning(f"ChromaDB directory not found: {config.CHROMA_DB_PATH}")
         print(f"\nWARNING: News database directory ('{config.CHROMA_DB_PATH}') not found.")
         print("         The system will attempt to create it, but it might be empty.")
         print("         Consider running: python fetch_news_data.py to populate it.")
    else:
         log.info(f"ChromaDB directory found: {config.CHROMA_DB_PATH}")


    # --- Load Dependencies ---
    log.info("Loading data sources...")
    data_sources.load_databases()
    log.info("Initializing vector DB connection (if needed by pipeline)...")


    # --- Process Input ---
    log.info(f"Loading farmer input data from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r') as f:
            input_json_data = json.load(f)

        adapted_data = adapt_input_data(input_json_data)

        log.info("Validating adapted input data using Pydantic model...")
        validated_input = models.InputParams.model_validate(adapted_data)
        log.info("Input data loaded and validated successfully.")

        # --- Run Pipeline ---
        log.info("Running recommendation pipeline...")
        # --- REMOVED explicit LLM initialization block ---
        # The explanation module now uses direct ollama calls, no shared object needed.

        recommendation_response = pipeline.run_recommendation_pipeline(validated_input)

        # --- Output Results ---
        log.info(f"Saving recommendation response to '{OUTPUT_FILE}'...")
        try:
            with open(OUTPUT_FILE, 'w') as f:
                f.write(recommendation_response.model_dump_json(indent=2))
            log.info(f"Recommendation successfully saved to {OUTPUT_FILE}.")
            print(f"\n✅ Output successfully written to: {OUTPUT_FILE}")

            if recommendation_response.recommendations:
                print("\n--- Top Recommendations ---")
                for i, rec in enumerate(recommendation_response.recommendations):
                    print(f"{i+1}. {rec.crop_name} (Score: {rec.recommendation_score})")
                print("\n--- LLM Generated Summary ---")
                print(recommendation_response.overall_llm_summary or "Summary not available.")
                print("---------------------------")
            else:
                print("\n⚠️ No suitable recommendations generated.")
                print(f"Reason/Context: {recommendation_response.weather_context_summary or recommendation_response.overall_llm_summary}")

        except Exception as e:
            log.error(f"Failed to save output JSON to {OUTPUT_FILE}: {e}", exc_info=True)
            print(f"\n❌ ERROR: Failed to write output file '{OUTPUT_FILE}'. Check permissions or disk space.")

    except FileNotFoundError:
        log.error(f"Input file not found: {INPUT_FILE}")
        print(f"\n❌ ERROR: Input file '{INPUT_FILE}' not found in the current directory.")
    except (ValidationError, ValueError) as e:
        log.error(f"Data validation or processing error: {e}", exc_info=False)
        print("\n❌ ERROR: Input data is invalid or cannot be processed.")
        print("\n--- Error Details ---")
        if isinstance(e, ValidationError):
            print("Input Data Validation Failed:")
            for error in e.errors():
                 loc_str = '.'.join(map(str, error['loc'])) if error['loc'] else 'N/A'
                 input_value = error.get('input', 'N/A')
                 print(f"  - Field: '{loc_str}'")
                 print(f"    Error: {error['msg']}")
                 print(f"    Input Value Type: {type(input_value).__name__}")
        else:
            print(f"Processing Error: {e}")
        print("-------------------")
    except Exception as e:
        log.critical(f"An unexpected critical error occurred in main: {e}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: An unexpected error occurred. Please check the logs.")
        print(f"Error Type: {type(e).__name__}")

    log.info("=============================================")
    log.info("=== Crop Recommendation Process Finished ===")
    log.info("=============================================")