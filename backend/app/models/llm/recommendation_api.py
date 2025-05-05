# recommendation_api.py
import json
import logging
import os
from typing import Union, Dict, Optional, Any

# Import necessary components from the project files
# These imports assume all .py files are in the same directory structure
# import config
import models.llm.models as models  # Import for InputParams and other models
import models.llm.pipeline as pipeline  # Import for pipeline function
import models.llm.data_sources as data_sources  # Import for DB initialization
import models.llm.fetch_news_data as fetch_news_data  # Import for vector DB initialization
from models.llm.main import adapt_input_data  # Reuse the adaptation logic from main.py
from pydantic import ValidationError

log = logging.getLogger(__name__)
# Configure logging if this file is run directly or imported early
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def get_recommendations(
    input_source: Union[Dict[str, Any], str],
) -> Optional[Dict[str, Any]]:
    """
    Generates crop recommendations based on the provided input JSON data.

    Args:
        input_source: Either a dictionary containing the parsed input data
                      (matching the structure expected by adapt_input_data, like ss.json)
                      OR a string representing the file path to the input JSON file.

    Returns:
        A dictionary representing the RecommendationResponse on success.
        Returns None if a critical error occurs during processing (e.g., file not found,
        validation error, pipeline failure). Check logs for details in case of None.
    """
    log.info("--- Received request for crop recommendation ---")
    recommendation_response_dict: Optional[Dict[str, Any]] = None

    try:
        # 1. Load Input Data
        raw_input_data: Dict[str, Any]
        if isinstance(input_source, str):
            file_path = input_source
            log.info(f"Input source is a file path: {file_path}")
            if not os.path.exists(file_path):
                log.error(f"Input file not found at path: {file_path}")
                return None  # Critical error: File not found
            with open(file_path, "r") as f:
                raw_input_data = json.load(f)
            log.info(f"Successfully loaded input data from file.")
        elif isinstance(input_source, dict):
            log.info("Input source is a dictionary.")
            raw_input_data = input_source
        else:
            log.error(
                f"Invalid input_source type: {type(input_source)}. Expected dict or str."
            )
            return None

        # 2. Adapt Input Data Structure
        log.info("Adapting input data structure...")
        adapted_data = adapt_input_data(raw_input_data)  # Reuse logic from main

        # 3. Validate Input Data
        log.info("Validating adapted input data...")
        validated_input = models.InputParams.model_validate(adapted_data)
        log.info("Input data validated successfully.")

        # 4. Initialize Dependencies (Load DBs, Models - ensures they are ready)
        # These can be potentially optimized if the function is called very frequently
        # by loading them once outside the function in the calling application.
        log.info("Loading/Verifying data sources and models...")
        data_sources.load_databases()
        if not data_sources.AGRONOMIC_DATA:
            log.error("Failed to load Agronomic Database. Cannot proceed.")
            return None  # Critical dependency missing
        if not fetch_news_data.initialize_vector_db_and_model():
            log.warning(
                "Failed to initialize vector DB/model. News features may be limited."
            )
            # Continue, but RAG might fail or return empty results

        # --- Ensure necessary modules using Ollama are ready ---
        # Currently, only explanation.py uses the direct 'ollama' library
        # No explicit object initialization needed here if using direct calls inside explanation.py

        # 5. Run the Recommendation Pipeline
        log.info("Running recommendation pipeline...")
        recommendation_response_obj = pipeline.run_recommendation_pipeline(
            validated_input
        )

        # 6. Prepare Output
        if recommendation_response_obj:
            log.info("Pipeline execution successful. Preparing response dictionary.")
            # Convert Pydantic model to dictionary for return
            recommendation_response_dict = recommendation_response_obj.model_dump(
                mode="json"
            )  # 'json' mode handles dates etc.
        else:
            # This case should ideally be handled within the pipeline returning a default error response
            log.error("Pipeline function returned None or an unexpected result.")
            # Attempt to create a basic error dictionary if possible
            request_details = models.RequestDetails(
                latitude=validated_input.latitude,
                longitude=validated_input.longitude,
                soil_type=validated_input.soil_type,
                land_size_acres=validated_input.land_size_acres,
                planting_date_estimate=validated_input.planting_date_estimate,
            ).model_dump(mode="json")
            recommendation_response_dict = {
                "request_details": request_details,
                "recommendations": [],
                "weather_context_summary": "Pipeline execution failed.",
                "news_headlines_considered": [],
                "news_impact_summary_llm": "Pipeline execution failed.",
                "overall_llm_summary": "Pipeline execution failed. Check logs.",
            }

    except FileNotFoundError as e:
        # This is handled above if input_source is a path, but added for completeness
        log.error(f"Input file error: {e}", exc_info=True)
        recommendation_response_dict = None
    except (ValidationError, ValueError) as e:
        log.error(f"Data validation or adaptation error: {e}", exc_info=True)
        # Consider returning specific validation errors if needed by the caller
        recommendation_response_dict = None
    except ImportError as e:
        log.critical(
            f"Import error: {e}. Ensure all required .py files are in the same directory or PYTHONPATH.",
            exc_info=True,
        )
        recommendation_response_dict = None  # Critical setup error
    except Exception as e:
        log.critical(
            f"An unexpected error occurred in get_recommendations: {e}", exc_info=True
        )
        recommendation_response_dict = None  # General critical error

    log.info("--- Crop recommendation request processing finished ---")
    return recommendation_response_dict
