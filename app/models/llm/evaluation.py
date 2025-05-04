# evaluation.py
import logging
from typing import List, Dict, Tuple, Optional  # Added Optional
import random

# Import directly (flat structure)
import models.llm.data_sources as data_sources
import models.llm.config as config
from models.llm.models import AgronomicInfo

log = logging.getLogger(__name__)

# --- Scoring Weights (DEMAND REMOVED, weights adjusted) ---
WEIGHTS = {
    "profitability": 0.55,  # Primary driver
    "subsidy": 0.30,  # Significant factor if present
    "risk": -0.15,  # Penalty for high risk
}
# --- Score Modifiers ---
TREND_MODIFIER = {
    "Rising": 1.2,
    "Slightly Rising": 1.1,
    "Stable": 1.0,
    "Falling": 0.8,
    "Volatile": 0.9,
}
COST_MODIFIER = {
    "Low": 1.2,
    "Low-Medium": 1.1,
    "Medium": 1.0,
    "Medium-High": 0.9,
    "High": 0.8,
}
# DEMAND_MODIFIER REMOVED
RISK_VOLATILITY_MODIFIER = {
    "High": 0.8,
    "Medium": 0.95,
    "Low": 1.05,
}  # Less penalty for low vol, higher for high
# Subsidy modifier applied directly in scoring logic


def check_critical_factors(
    suitable_crops: List[str],
    events: List[Dict],
    subsidies: List[Dict],  # Use the raw dicts from input adaptation
    latitude: float,
    longitude: float,
) -> Dict[str, Dict]:
    """
    Identifies deal-breakers (elimination) or game-changers (high priority)
    based on critical events and high-impact subsidies.

    Returns:
        Dict mapping crop_name to status info:
        {"status": "ok" | "eliminated_event" | "high_priority_subsidy",
         "reason": str | None,
         "priority_subsidy": Dict | None}
    """
    log.info("--- Checking Critical Factors (Hard Cutoffs & Game Changers) ---")
    region = config.get_region_code(latitude, longitude)
    log.info(f"Operating in derived region: {region}")
    crop_status = {
        crop: {"status": "ok", "reason": None, "priority_subsidy": None}
        for crop in suitable_crops
    }

    # 1. Check for Critical Events (Elimination Factor)
    log.info(f"Analyzing {len(events)} relevant events for critical impact...")
    eliminated_count = 0
    for event in events:
        severity = event.get("severity", "medium").lower()
        event_type = event.get("type", "unknown").lower()
        affected_crops = event.get("crop_affected", [])  # Can be empty (regional event)
        summary = event.get("summary", "N/A")
        event_regions = event.get("region_codes", [])  # Regions event applies to

        # Check if event applies to current region
        applies_to_region = not event_regions or region in event_regions

        # Define what constitutes a critical event (e.g., high severity, specific keywords)
        is_critical_event = severity == "high" or any(
            k in summary.lower()
            for k in ["ban", "severe outbreak", "major damage", "quarantine"]
        )

        if applies_to_region and is_critical_event:
            log.warning(
                f"CRITICAL EVENT DETECTED in region {region}: {summary} (Severity: {severity})"
            )
            # If event affects specific crops, eliminate them. If affects all (empty list), eliminate all in region.
            crops_to_eliminate = (
                affected_crops if affected_crops else suitable_crops
            )  # If empty list means applies to all

            for crop in crops_to_eliminate:
                if crop in crop_status and crop_status[crop]["status"] == "ok":
                    log.warning(
                        f"-> CRITICAL CUTOFF: Eliminating '{crop}' due to event: {summary}"
                    )
                    crop_status[crop]["status"] = "eliminated_event"
                    crop_status[crop]["reason"] = (
                        f"Critical Event ({severity}): {summary}"
                    )
                    eliminated_count += 1

    if eliminated_count > 0:
        log.info(f"Eliminated {eliminated_count} crops due to critical events.")

    # 2. Check for High-Impact Subsidies (Game Changer Factor)
    log.info(f"Analyzing {len(subsidies)} available subsidies for game-changers...")
    game_changer_count = 0
    for crop in suitable_crops:
        # Skip if already eliminated
        if crop_status[crop]["status"] != "ok":
            continue

        # Get subsidies relevant to this specific crop and region
        # Note: get_subsidy_details_for_crop now just filters the provided list
        relevant_subsidies = data_sources.get_subsidy_details_for_crop(
            crop, subsidies, region
        )

        agronomic_info: Optional[AgronomicInfo] = data_sources.get_crop_agronomic_data(
            crop
        )
        if not agronomic_info:
            continue  # Should not happen if suitable_crops list is correct

        input_cost_cat = agronomic_info.input_cost_cat
        rough_cost = config.ROUGH_INPUT_COST_ESTIMATES.get(
            input_cost_cat, 20000
        )  # Use estimate from config

        for sub in relevant_subsidies:
            estimated_value = sub.get("estimated_value_inr")
            benefit_summary = sub.get("benefit_summary", "").lower()
            is_game_changer = False

            # Define criteria for a "game-changer" subsidy (adjust thresholds as needed)
            # Example 1: High percentage cost reduction mentioned
            if (
                "% cost reduction" in benefit_summary
                or "% subsidy on input" in benefit_summary
            ):
                try:
                    # Extract percentage (simple extraction, might need refinement)
                    percentage_str = benefit_summary.split("%")[0].split()[-1]
                    percentage = float(percentage_str)
                    if (
                        percentage >= config.HARD_CUTOFF_SUBSIDY_THRESHOLD * 100
                    ):  # e.g., 80% or more
                        is_game_changer = True
                        log.debug(
                            f"Game changer identified for '{crop}': High percentage subsidy ({percentage}%) found in '{sub['program']}'"
                        )
                except ValueError:
                    pass  # Ignore if percentage cannot be parsed

            # Example 2: Estimated value is a large fraction of estimated input cost
            elif (
                estimated_value
                and rough_cost > 0
                and (estimated_value / rough_cost)
                >= config.HARD_CUTOFF_SUBSIDY_THRESHOLD
            ):
                is_game_changer = True
                log.debug(
                    f"Game changer identified for '{crop}': Estimated subsidy value (₹{estimated_value}) >= {config.HARD_CUTOFF_SUBSIDY_THRESHOLD * 100}% of estimated cost (₹{rough_cost}) for '{sub['program']}'"
                )

            # Example 3: Specific keywords indicating full coverage
            elif any(
                k in benefit_summary
                for k in ["free seed", "100% subsidy", "full cost coverage"]
            ):
                is_game_changer = True
                log.debug(
                    f"Game changer identified for '{crop}': Keyword indicates full coverage in '{sub['program']}'"
                )

            if is_game_changer:
                # If a game-changer is found, mark the crop and store the subsidy info
                log.info(
                    f"--> GAME CHANGER: Marking '{crop}' as high priority due to subsidy: {sub.get('program', 'N/A')}"
                )
                crop_status[crop]["status"] = "high_priority_subsidy"
                crop_status[crop]["priority_subsidy"] = (
                    sub  # Store the specific subsidy dict
                )
                game_changer_count += 1
                break  # Only need one game-changer per crop

    if game_changer_count > 0:
        log.info(f"Identified {game_changer_count} crops with high-priority subsidies.")

    log.info("Finished checking critical factors.")
    return crop_status


def calculate_scores(
    candidate_crops: List[str],
    crop_details_data: Dict,
    critical_status: Dict,
    retrieved_news_context: List[Dict],  # Use metadata from retrieved news
) -> List[Tuple[str, float, Dict]]:
    """
    Calculates a weighted score for each candidate crop based on
    Profitability, Subsidy impact, and Risk.

    Args:
        candidate_crops: List of crops remaining after critical checks.
        crop_details_data: Dict containing agronomic info, forecasts etc.
        critical_status: Dict with status ("ok", "high_priority_subsidy").
        retrieved_news_context: List of news item dicts (metadata) from ChromaDB.

    Returns:
        List of tuples: (crop_name, normalized_score, scoring_factors_dict), sorted descending by score.
    """
    log.info(f"--- Calculating Scores for {len(candidate_crops)} candidates ---")
    scores = []

    # --- Simplified News Sentiment Calculation (based on retrieved headlines) ---
    # This is a basic heuristic, LLM analysis in explanation.py provides deeper insight
    news_sentiment_modifier = 0.0  # Start neutral
    news_headlines = [
        item.get("headline", "").lower() for item in retrieved_news_context
    ]
    news_full_text = " ".join(news_headlines)  # Combine headlines for keyword search

    # Positive keywords (increase score slightly)
    positive_keywords = [
        "export rise",
        "demand strong",
        "trade deal",
        "good monsoon",
        "msp hike",
        "price increase",
    ]
    if any(k in news_full_text for k in positive_keywords):
        news_sentiment_modifier += 0.1

    # Negative keywords (decrease score more significantly)
    negative_keywords = [
        "ban",
        "concern",
        "pest outbreak",
        "drought",
        "poor monsoon",
        "import surge",
        "price fall",
        "farmer protest",
    ]
    if any(k in news_full_text for k in negative_keywords):
        news_sentiment_modifier -= 0.15  # Higher penalty for negative news

    log.info(
        f"Calculated simple news sentiment modifier based on {len(news_headlines)} headlines: {news_sentiment_modifier:+.2f}"
    )

    for crop in candidate_crops:
        status_info = critical_status.get(crop, {})
        # Skip if eliminated (though candidate_crops should already be filtered)
        if status_info.get("status") == "eliminated_event":
            continue

        data = crop_details_data.get(crop, {})
        if not data:
            log.warning(
                f"Missing detailed data for candidate crop '{crop}'. Skipping scoring."
            )
            continue

        agronomic_info: Optional[AgronomicInfo] = data.get("agronomic_info")
        if not agronomic_info:
            log.warning(
                f"Missing agronomic info for candidate crop '{crop}'. Skipping scoring."
            )
            continue

        # --- 1. Profitability Score ---
        price_forecast = data.get("price_forecast", {})
        cost_category = agronomic_info.input_cost_cat
        yield_factor = (
            1.0  # Base yield factor (could adjust based on yield estimate range later)
        )

        # Base profitability = Yield * Price Trend Modifier * Inverse Cost Modifier
        profit_score = (
            yield_factor
            * TREND_MODIFIER.get(price_forecast.get("trend", "Stable"), 1.0)
            * COST_MODIFIER.get(cost_category, 1.0)
        )
        profit_details = f"YieldFactor:{yield_factor:.1f} * PriceTrend:{price_forecast.get('trend', 'N/A')} ({TREND_MODIFIER.get(price_forecast.get('trend', 'Stable'), 1.0):.1f}) * InvCostFactor:{cost_category} ({COST_MODIFIER.get(cost_category, 1.0):.1f})"

        # --- 2. Subsidy Score ---
        subsidy_score = 1.0  # Base score (no subsidy impact)
        subsidy_details = "Standard or None"
        priority_subsidy = status_info.get("priority_subsidy")

        if priority_subsidy:
            # Significant boost if a game-changer subsidy was identified
            subsidy_score = 1.6  # Higher multiplier for game-changer
            subsidy_details = (
                f"High Priority Game-Changer: {priority_subsidy.get('program', 'N/A')}"
            )
        elif status_info.get("status") == "ok":
            # Check if *any* standard relevant subsidies were found (even if not game-changers)
            # We need the filtered list again, or assume `get_subsidy_details_for_crop` was called before
            # Re-calling it might be slightly inefficient but ensures consistency.
            # This requires passing the original input subsidies list down here.
            # **Simplification**: For now, let's assume standard subsidies give a smaller boost if no game-changer.
            # A better approach would be to pass the filtered standard subsidies list here.
            # if data.get("has_standard_subsidies"): # Assuming this flag was set earlier
            #     subsidy_score = 1.1
            #     subsidy_details = f"Relevant standard subsidies found."
            pass  # No change for standard subsidies for now without passing the list

        # --- 3. Risk Score ---
        price_volatility = price_forecast.get("volatility", "Medium")
        # Base risk from price volatility
        volatility_risk_score = RISK_VOLATILITY_MODIFIER.get(price_volatility, 1.0)
        # Combine with news sentiment
        # Risk score: Lower is better. Start from volatility base, add penalty from news.
        # Let's define risk score where 1.0 is neutral. >1 is higher risk, <1 is lower risk.
        # Invert volatility modifier: High Vol -> Higher risk score (e.g., 1.2)
        # Invert news modifier: Negative news -> Higher risk score
        inverted_volatility_mod = 1.0 + (
            1.0 - RISK_VOLATILITY_MODIFIER.get(price_volatility, 1.0)
        )  # e.g. High Vol (0.8) -> 1 + (1-0.8) = 1.2
        risk_score = (
            inverted_volatility_mod - news_sentiment_modifier
        )  # Subtract positive sentiment (reduces risk score), add negative (increases risk score)

        risk_details = f"Base Volatility: {price_volatility} (Mod: {inverted_volatility_mod:.2f}), News Sentiment Mod: {news_sentiment_modifier:+.2f}"

        # --- Weighted Final Score ---
        # Ensure weights sum reasonably (or normalize if needed, though not strictly necessary here)
        final_score_components = {
            "profitability": profit_score * WEIGHTS["profitability"],
            "subsidy": subsidy_score * WEIGHTS["subsidy"],
            "risk": (risk_score - 1.0)
            * WEIGHTS["risk"],  # Apply risk weight to deviation from neutral (1.0)
        }
        final_score_raw = sum(final_score_components.values())

        # Store factors used for explanation
        scoring_factors = {
            "profitability": {"score": profit_score, "details": profit_details},
            "subsidy": {"score": subsidy_score, "details": subsidy_details},
            "risk": {"score": risk_score, "details": risk_details},
        }

        log.info(f"  Scores - {crop}:")
        log.info(
            f"    Profitability: {profit_score:.2f} (Weighted: {final_score_components['profitability']:.3f})"
        )
        log.info(
            f"    Subsidy:       {subsidy_score:.2f} (Weighted: {final_score_components['subsidy']:.3f})"
        )
        log.info(
            f"    Risk:          {risk_score:.2f} (Weighted Component: {final_score_components['risk']:.3f})"
        )
        log.info(f"    => Raw Total:   {final_score_raw:.3f}")

        scores.append((crop, final_score_raw, scoring_factors))

    # --- Sort and Normalize Scores ---
    scores.sort(key=lambda item: item[1], reverse=True)  # Sort by raw score descending

    if not scores:
        log.warning("No scores calculated for any candidate crop.")
        return []

    # Normalize scores to a 0-100 scale (or similar) for better interpretation
    min_score = scores[-1][1]
    max_score = scores[0][1]
    score_range = max_score - min_score

    normalized_scores = []
    log.info("--- Normalizing Scores (0-100 scale) ---")
    for crop, score, factors in scores:
        if (
            score_range <= 0.001
        ):  # Avoid division by zero or near-zero if all scores are identical
            norm_score = 50.0 + random.uniform(-0.1, 0.1)  # Assign a mid-range score
        else:
            # Linear scaling to ~10-95 range, add slight randomness
            scaled_score = (score - min_score) / score_range  # Scale to 0-1
            norm_score = 10 + scaled_score * 85  # Map to 10-95 range
            # Add small random jitter for tie-breaking / realism
            norm_score += random.uniform(-1.5, 1.5)
            norm_score = max(0, min(100, norm_score))  # Clamp to 0-100

        log.info(f"  Normalized Score - {crop}: {norm_score:.1f} (Raw: {score:.3f})")
        normalized_scores.append((crop, norm_score, factors))

    log.info(f"Finished scoring and ranking {len(normalized_scores)} crops.")
    return normalized_scores
