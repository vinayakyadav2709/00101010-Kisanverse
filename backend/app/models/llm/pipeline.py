# pipeline.py
import logging
import os  # Added for makedirs
from datetime import datetime, date  # Import date
from typing import List, Dict, Optional

# ChromaDB & Embeddings (Still needed for RAG)
import chromadb
from sentence_transformers import SentenceTransformer

# Import pipeline components and models directly
import models.llm.data_sources as data_sources
import models.llm.filtering as filtering
import models.llm.evaluation as evaluation
import models.llm.explanation as explanation  # Still needed for non-LLM functions + new LLM functions
import models.llm.config as config  # Configuration for DB paths, etc.
from models.llm.models import (
    InputParams,
    RecommendationResponse,
    RequestDetails,
    Recommendation,
    KeyMetrics,
    RelevantSubsidyOutput,
    PlottingData,
    AgronomicInfo,
    SubsidyInfo,
    WeatherInfo,
    PesticideSuggestion,
    NewsItemReference,
    PriceChartData,
    WaterChartData,
    FertilizerChartData,
    PriceDataPoint,
    WaterDataPoint,
    FertilizerDataPoint,
)

log = logging.getLogger(__name__)

# --- RAG Helper Function (Uses SentenceTransformer for embedding, Chroma for storage/retrieval) ---
rag_chroma_client: Optional[chromadb.ClientAPI] = None
rag_collection: Optional[chromadb.Collection] = None
rag_embedding_model: Optional[SentenceTransformer] = None


def retrieve_relevant_news_from_chroma(query_text: str, num_results: int) -> List[Dict]:
    """
    Embeds a query using SentenceTransformer and retrieves relevant news items
    (metadata) from ChromaDB. Handles initialization.
    """
    global rag_chroma_client, rag_collection, rag_embedding_model
    log.info(
        f"Attempting to retrieve {num_results} relevant news items for query: '{query_text[:100]}...'"
    )

    # Initialize ChromaDB and embedding model if not already done
    if not rag_collection or not rag_embedding_model:
        log.info("Initializing ChromaDB client/collection/model for RAG retrieval...")
        try:
            if not rag_chroma_client:
                os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
                rag_chroma_client = chromadb.PersistentClient(
                    path=config.CHROMA_DB_PATH
                )
                log.info(
                    f"ChromaDB PersistentClient initialized at {config.CHROMA_DB_PATH}"
                )

            if not rag_collection:
                rag_collection = rag_chroma_client.get_or_create_collection(
                    name=config.CHROMA_COLLECTION_NAME
                )
                log.info(
                    f"Got Chroma collection '{config.CHROMA_COLLECTION_NAME}' ({rag_collection.count()} items)"
                )

            if not rag_embedding_model:
                # Load the SentenceTransformer model specified in config
                rag_embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
                log.info(
                    f"Embedding model '{config.EMBEDDING_MODEL_NAME}' loaded for RAG."
                )

        except Exception as e:
            log.error(
                f"Failed to initialize components for RAG retrieval: {e}", exc_info=True
            )
            return []

    # Perform the query
    try:
        # Generate embedding using the loaded SentenceTransformer model
        query_embedding = rag_embedding_model.encode(query_text).tolist()

        # Query ChromaDB using the generated embedding
        results = rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            include=["metadatas", "documents", "distances"],  # Include relevant fields
        )

        # Process results
        retrieved_items = []
        if results and results.get("metadatas") and results.get("distances"):
            metadatas_list = results["metadatas"][0]
            distances_list = results["distances"][0]
            documents_list = results.get("documents", [[]])[
                0
            ]  # Get document snippets if included

            for i, meta in enumerate(metadatas_list):
                item_data = dict(meta)  # Copy metadata
                item_data["distance"] = distances_list[i]  # Add similarity distance
                if i < len(documents_list):  # Add document snippet if available
                    item_data["document_snippet"] = documents_list[i][:150]
                retrieved_items.append(item_data)

            log.info(
                f"Retrieved {len(retrieved_items)} news items from ChromaDB with distances."
            )
            return retrieved_items
        else:
            log.warning("ChromaDB query returned no results or unexpected format.")
            return []

    except Exception as e:
        log.error(f"Error during ChromaDB query or embedding: {e}", exc_info=True)
        return []


# --- Main Pipeline Function ---
def run_recommendation_pipeline(input_data: InputParams) -> RecommendationResponse:
    """Orchestrates the full recommendation pipeline using RAG for news context and direct Ollama calls."""
    start_time = datetime.now()
    log.info(f"--- Starting Recommendation Pipeline Run: {start_time.isoformat()} ---")

    # 0. Prepare Request Details & Default Response Structure
    request_details = RequestDetails(
        latitude=input_data.latitude,
        longitude=input_data.longitude,
        soil_type=input_data.soil_type,
        land_size_acres=input_data.land_size_acres,
        planting_date_estimate=input_data.planting_date_estimate,
        timestamp=start_time,
    )
    response = RecommendationResponse(
        request_details=request_details,
        recommendations=[],
        weather_context_summary="Analysis did not complete.",
        news_headlines_considered=[],
        news_impact_summary_llm="News analysis not performed.",
        overall_llm_summary="Pipeline did not complete successfully.",
    )

    try:
        # 1. Verify Data Sources
        log.info("Verifying data sources...")
        if not data_sources.AGRONOMIC_DATA:
            raise ValueError(
                "Agronomic database is missing or empty. Run fetch_agronomic_data.py."
            )
        log.info(
            f"Agronomic database loaded with {len(data_sources.AGRONOMIC_DATA)} crops."
        )

        # 2. Get Potential Crops
        potential_crops = data_sources.get_potential_crops(
            input_data.latitude, input_data.longitude
        )
        if not potential_crops:
            response.overall_llm_summary = (
                "No potential crops identified for this region."
            )
            return response
        log.info(f"Identified {len(potential_crops)} potential crops initially.")

        # 3. Fetch Supporting Data
        crop_details_data = {}
        valid_potential_crops = []
        log.info(
            "Fetching supporting agronomic details, forecasts, and simulated events..."
        )
        for crop in potential_crops:
            agronomic_info = data_sources.get_crop_agronomic_data(crop)
            if not agronomic_info:
                continue
            valid_potential_crops.append(crop)
            crop_details_data[crop] = {
                "agronomic_info": agronomic_info,
                "yield_estimate": data_sources.get_yield_estimate(
                    crop, input_data.soil_type
                ),
                "price_forecast": data_sources.get_price_forecast(crop),
                "input_cost_category": agronomic_info.input_cost_cat,
            }
        relevant_events = data_sources.get_relevant_events(
            input_data.latitude, input_data.longitude, valid_potential_crops
        )
        input_subsidy_dicts = [
            s.model_dump(exclude_none=True) for s in input_data.available_subsidies
        ]

        # 4. Agronomic Filtering
        log.info("Running Agronomic Filtering...")
        agronomically_suitable = filtering.filter_agronomically_suitable_crops(
            valid_potential_crops, input_data
        )
        if not agronomically_suitable:
            log.warning(
                "No crops passed basic agronomic suitability checks (soil/water)."
            )
            response.weather_context_summary = "No crops found suitable based on soil type and forecast water availability."
            response.overall_llm_summary = "Could not find suitable crops matching your soil and local weather forecast."
            return response
        log.info(f"{len(agronomically_suitable)} crops passed agronomic filtering.")

        # --- RAG Step: Retrieve Relevant News ---
        log.info("Retrieving relevant news context using RAG...")
        region_code = config.get_region_code(input_data.latitude, input_data.longitude)
        rag_query = (
            f"Recent agriculture news, policies, market trends, weather impacts, and events relevant to India, "
            f"specifically considering region {region_code} and potential crops like {', '.join(agronomically_suitable[:5])}..."
        )
        retrieved_news_items = retrieve_relevant_news_from_chroma(
            rag_query, config.NUM_RELEVANT_NEWS_TO_RETRIEVE
        )
        top_headlines_for_llm = [
            item.get("headline", "N/A") for item in retrieved_news_items
        ]
        response.news_headlines_considered = top_headlines_for_llm
        log.info(f"Retrieved {len(retrieved_news_items)} news items via RAG.")

        # --- Ollama Analysis of News Context ---
        log.info(
            "Performing Ollama analysis for news impact summary and crop relevance mapping..."
        )
        # Call the function in explanation.py which now uses direct ollama calls
        news_impact_summary, news_mapping_by_headline = (
            explanation.analyze_news_context_with_llm(
                top_headlines_for_llm, agronomically_suitable
            )
        )
        response.news_impact_summary_llm = news_impact_summary
        log.info(f"Ollama News Impact Summary: {news_impact_summary}")

        # 5. Critical Factor Checks
        log.info("Checking Critical Factors (Events, Game-Changer Subsidies)...")
        current_region_code = config.get_region_code(
            input_data.latitude, input_data.longitude
        )
        critical_status = evaluation.check_critical_factors(
            agronomically_suitable,
            relevant_events,
            input_subsidy_dicts,
            input_data.latitude,
            input_data.longitude,
        )
        candidate_crops = [
            crop
            for crop in agronomically_suitable
            if critical_status.get(crop, {}).get("status") != "eliminated_event"
        ]
        if not candidate_crops:
            log.warning(
                "All agronomically suitable crops were eliminated by critical factors (events)."
            )
            elimination_reasons = "; ".join(
                set(
                    cs.get("reason", "N/A")
                    for cs in critical_status.values()
                    if cs.get("status") == "eliminated_event"
                )
            )
            summary = f"All suitable crops ruled out due to critical events. Reasons: {elimination_reasons}"
            response.weather_context_summary = summary
            response.overall_llm_summary = summary
            return response
        log.info(
            f"{len(candidate_crops)} crops remain as candidates after critical checks."
        )

        # 6. Scoring and Ranking
        log.info("Calculating Scores and Ranking candidates...")
        scored_ranked_crops = evaluation.calculate_scores(
            candidate_crops, crop_details_data, critical_status, retrieved_news_items
        )
        if not scored_ranked_crops:
            log.warning("Scoring process yielded no ranked crops.")
            response.overall_llm_summary = (
                "Failed to calculate scores for candidate crops."
            )
            return response

        # 7. Generate Detailed Recommendations
        log.info(
            f"Generating detailed recommendations for top {config.MAX_RECOMMENDATIONS} crops..."
        )
        recommendations_output: List[Recommendation] = []
        rank = 1
        for crop_name, norm_score, factors in scored_ranked_crops:
            if rank > config.MAX_RECOMMENDATIONS:
                break
            log.info(
                f"--- Generating details for Rank {rank}: {crop_name} (Score: {norm_score:.1f}) ---"
            )

            # Fetch necessary data
            crop_data = crop_details_data[crop_name]
            status_info = critical_status[crop_name]
            agronomic_info: Optional[AgronomicInfo] = crop_data.get("agronomic_info")
            if not agronomic_info:
                continue
            region = config.get_region_code(input_data.latitude, input_data.longitude)

            # Subsidies, Metrics, Risks, Pesticides, Explanations
            relevant_subsidy_list = data_sources.get_subsidy_details_for_crop(
                crop_name, input_subsidy_dicts, region
            )
            output_subsidies = [
                RelevantSubsidyOutput(**s) for s in relevant_subsidy_list
            ]
            metrics = KeyMetrics(
                expected_yield_range=crop_data["yield_estimate"],
                price_forecast_trend=crop_data["price_forecast"].get("trend", "N/A"),
                estimated_input_cost_category=agronomic_info.input_cost_cat,
                primary_fertilizer_needs=agronomic_info.fertilizer_needs,
            )
            risks = explanation.get_primary_risks(
                crop_name, factors, relevant_events, region
            )
            pesticides: List[PesticideSuggestion] = data_sources.recommend_pesticides(
                crop_name, risks
            )
            expl_points = explanation.generate_explanation(
                crop_name,
                norm_score,
                factors,
                {"soil_type": input_data.soil_type},
                status_info,
            )

            # Plotting Data Generation
            log.info(f"[{crop_name}] Generating plotting data...")
            plotting_data_obj = PlottingData(
                price_forecast_chart=PriceChartData(
                    description="Price Forecast (Simulated)", data=[]
                ),
                water_need_chart=WaterChartData(
                    description="Relative Water Need by Stage", data=[]
                ),
                fertilizer_schedule_chart=FertilizerChartData(
                    description="Typical Fertilizer Timing", data=[]
                ),
            )
            try:
                price_chart_raw = data_sources.generate_price_chart_data(
                    crop_name, crop_data["price_forecast"]
                )
                water_chart_raw = data_sources.generate_water_chart_data(crop_name)
                fert_chart_raw = data_sources.generate_fertilizer_chart_data(crop_name)

                price_data_points = [PriceDataPoint(**p) for p in price_chart_raw]
                water_data_points = [WaterDataPoint(**w) for w in water_chart_raw]
                fert_data_points = [FertilizerDataPoint(**f) for f in fert_chart_raw]

                plotting_data_obj = PlottingData(
                    price_forecast_chart=PriceChartData(
                        description="Predicted Price Range (INR/Quintal, Simulated)",
                        data=price_data_points,
                    ),
                    water_need_chart=WaterChartData(
                        description="Relative Water Requirement by Growth Stage (1-5)",
                        data=water_data_points,
                    ),
                    fertilizer_schedule_chart=FertilizerChartData(
                        description="Typical Nutrient Application Timing",
                        data=fert_data_points,
                    ),
                )
                log.info(
                    f"[{crop_name}] PlottingData created successfully with specific models."
                )
            except Exception as e:
                log.error(
                    f"Failed to generate or validate Plotting Data for {crop_name}!",
                    exc_info=True,
                )
                log.warning(
                    f"[{crop_name}] Using empty plotting data due to generation error."
                )

            # Map Relevant News
            crop_specific_mapped_headlines = news_mapping_by_headline.get(crop_name, [])
            crop_news_references = []
            if crop_specific_mapped_headlines:
                log.info(
                    f"[{crop_name}] Found {len(crop_specific_mapped_headlines)} relevant headlines from Ollama mapping. Matching with RAG results..."
                )
                for headline in crop_specific_mapped_headlines:
                    found_item = next(
                        (
                            item
                            for item in retrieved_news_items
                            if item.get("headline") == headline
                        ),
                        None,
                    )
                    if found_item:
                        crop_news_references.append(
                            NewsItemReference(
                                headline=headline,
                                url=found_item.get("url"),
                                source=found_item.get("source"),
                                date=found_item.get("date"),
                            )
                        )
                    else:
                        log.warning(
                            f"  - Headline '{headline[:50]}...' mapped by Ollama but not found in RAG results list?"
                        )
            else:
                log.info(f"[{crop_name}] No specific news headlines mapped by Ollama.")

            # Assemble Recommendation
            recommendations_output.append(
                Recommendation(
                    rank=rank,
                    crop_name=crop_name,
                    recommendation_score=round(norm_score, 1),
                    explanation_points=expl_points,
                    key_metrics=metrics,
                    relevant_subsidies=output_subsidies,
                    primary_risks=risks,
                    suggested_pesticides=pesticides,
                    plotting_data=plotting_data_obj,
                    relevant_news=crop_news_references,
                )
            )
            rank += 1

        # 8. Final Response Assembly
        # Generate Weather Summary
        weather_summary = "Weather forecast data analyzed."
        if input_data.weather_forecast:
            try:
                num_days = len(input_data.weather_forecast)
                if num_days > 0:
                    avg_temp = (
                        sum(
                            (wp.temp_max + wp.temp_min) / 2
                            for wp in input_data.weather_forecast
                        )
                        / num_days
                    )
                    total_rain = sum(wp.precip for wp in input_data.weather_forecast)
                    weather_summary = f"Forecast ({num_days} days) indicates avg temp ~{avg_temp:.1f}°C, total rainfall ~{total_rain:.1f}mm."
                else:
                    weather_summary = "Weather forecast data was empty."
            except Exception as e:
                weather_summary = f"Error processing weather forecast data: {e}"
                log.error("Weather summary calculation failed", exc_info=True)
        response.weather_context_summary = weather_summary

        # Add recommendations
        response.recommendations = recommendations_output

        # Generate Final Ollama Summary
        if response.recommendations:
            log.info("Generating final Ollama summary...")
            # Call the function in explanation.py which now uses direct ollama calls
            llm_final_summary = explanation.generate_llm_summary(
                response.recommendations,
                response.news_impact_summary_llm
                or "Not analyzed.",  # Use the stored Ollama news summary
                response.weather_context_summary,
            )
            response.overall_llm_summary = llm_final_summary
        else:
            # Create summary if no recommendations
            if not agronomically_suitable:
                response.overall_llm_summary = "Based on your soil type and the weather forecast, no suitable crops were identified for your location at this time."
            elif not candidate_crops:
                response.overall_llm_summary = "While some crops were initially suitable, critical factors (like severe weather or pest alerts) ruled them out."
            else:
                response.overall_llm_summary = "No crop recommendations could be generated after the full analysis."

        log.info(
            f"--- Pipeline Finished Successfully at {datetime.now().isoformat()} ---"
        )
        return response

    except Exception as e:
        log.critical(
            f"--- Pipeline Failed Critically at {datetime.now().isoformat()} ---",
            exc_info=True,
        )
        response.overall_llm_summary = f"An unexpected error occurred during processing ({type(e).__name__}). Please check logs or contact support."
        response.weather_context_summary = "Analysis failed due to an internal error."
        response.recommendations = []
        return response
