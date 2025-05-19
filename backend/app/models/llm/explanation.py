# explanation.py
import logging
import json
from typing import List, Dict, Optional, Tuple, Any, Union, Generator
import re

# --- MODIFIED: Use direct ollama library ---
import ollama
# --- END MODIFIED ---

# LangChain Imports REMOVED: ChatOllama, ChatPromptTemplate, StrOutputParser, JsonOutputParser, OutputParserException

# Import project modules
import models.llm.data_sources as data_sources  # To get crop agronomic data
from models.llm.models import (
    ExplanationPoint,
    Recommendation,
    AgronomicInfo,
    PesticideSuggestion,
    NewsItemReference,
)
from models.llm.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)  # Keep config imports

log = logging.getLogger(__name__)

# --- REMOVED Global LLM Initialization ---
# The 'llm' object from LangChain is no longer used here.


# --- LLM for News Context Analysis (Using direct ollama call) ---
def analyze_news_context_with_llm(
    top_news_headlines: List[str], candidate_crops: List[str]
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Uses direct ollama library call to get a summary of news impact and map relevant headlines to crops.

    Args:
        top_news_headlines: List of the most relevant news headlines retrieved via RAG.
        candidate_crops: List of crop names being considered after initial filtering.

    Returns:
        A tuple containing:
        - overall_impact_summary (str): LLM-generated summary of the news context.
        - news_mapping (Dict[str, List[str]]): Dictionary mapping crop names to list of relevant headlines.
    """
    log.info(
        f"Analyzing {len(top_news_headlines)} top news headlines with direct Ollama call for relevance to {len(candidate_crops)} crops..."
    )
    default_impact_summary = "Ollama news analysis call failed or returned no summary."
    news_mapping = {crop: [] for crop in candidate_crops}

    if not top_news_headlines:
        log.warning("No news headlines provided for Ollama analysis.")
        return "No relevant news headlines were available for analysis.", news_mapping
    if not candidate_crops:
        log.warning("No candidate crops provided for Ollama news relevance analysis.")
        return "No candidate crops specified for news relevance mapping.", news_mapping

    # Construct prompt manually for direct Ollama call
    headlines_numbered = "\n".join(
        f"{i}. {h}" for i, h in enumerate(top_news_headlines)
    )
    crop_list_str = ", ".join(candidate_crops)

    system_prompt = (
        "You are an expert agricultural analyst specializing in the Indian context. "
        "Your task is to analyze recent news headlines related to Indian agriculture and determine their potential relevance to specific crops. "
        "Focus on policy changes, market trends (prices, export/import), weather impacts (monsoon, drought), pest/disease outbreaks, subsidies, MSP, and farmer issues. "
        "Provide a concise overall summary of the current news climate for agriculture based on these headlines. "
        "Then, identify which headlines are most relevant to EACH of the provided candidate crops. "
        "Output *only* the requested JSON object, nothing else before or after."
    )
    user_prompt = (
        f"Candidate Crops: {crop_list_str}\n\n"
        f"Recent Headlines (indexed):\n{headlines_numbered}\n\n"
        "Based ONLY on the headlines provided, generate a JSON object with two keys:\n"
        "1. 'overall_summary': A brief (2-3 sentences) summary of the general sentiment or key themes emerging from these headlines regarding Indian agriculture.\n"
        "2. 'crop_relevance': A dictionary where keys are the EXACT crop names from the 'Candidate Crops' list. The value for each crop key should be a list of the index numbers (integers) of the headlines that are potentially relevant to that specific crop. Include a headline index if it mentions the crop, its inputs (like specific fertilizers), major growing regions indirectly, or significant market factors affecting it. If no headlines are relevant to a crop, provide an empty list [].\n\n"
        "Example relevant mentions: 'MSP increased for Paddy', 'Fall Armyworm alert impacting Maize', 'Cotton export policy revised', 'Fertilizer subsidy changes announced', 'Drought conditions affect Kharif sowing'.\n\n"
        'Format output STRICTLY as JSON: {{"overall_summary": "Your summary.", "crop_relevance": {{"Crop Name 1": [index1, index2], "Crop Name 2": [], ...}} }}\n'
        "JSON Output:"
    )

    try:
        log.info("Invoking ollama.chat for news analysis (expecting JSON)...")
        # Use the ollama client configured with the base URL from config
        client = ollama.Client(host=OLLAMA_BASE_URL)  # Initialize client with host URL
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",  # Request JSON output format
            options={"temperature": LLM_TEMPERATURE},
        )

        log.debug(f"Ollama Raw Response: {response}")

        # Parse JSON response manually
        if response and "message" in response and "content" in response["message"]:
            raw_content = response["message"]["content"]
            log.debug(f"Ollama Raw Content (JSON string): {raw_content}")
            try:
                llm_result = json.loads(raw_content)
                log.debug(f"Parsed LLM Analysis Dict: {llm_result}")

                if isinstance(llm_result, dict):
                    impact_summary = llm_result.get(
                        "overall_summary", default_impact_summary
                    )
                    crop_relevance_map = llm_result.get("crop_relevance", {})

                    # Populate news_mapping from parsed results
                    for crop_name, headline_indices in crop_relevance_map.items():
                        if crop_name in news_mapping:
                            if isinstance(headline_indices, list):
                                headlines_for_crop = []
                                for idx in headline_indices:
                                    if isinstance(idx, int) and 0 <= idx < len(
                                        top_news_headlines
                                    ):
                                        headlines_for_crop.append(
                                            top_news_headlines[idx]
                                        )
                                    else:
                                        log.warning(
                                            f"Ollama returned invalid index '{idx}' for crop '{crop_name}'."
                                        )
                                news_mapping[crop_name] = headlines_for_crop[
                                    :3
                                ]  # Limit to 3
                            else:
                                log.warning(
                                    f"Ollama returned non-list indices '{headline_indices}' for crop '{crop_name}'."
                                )
                        else:
                            log.warning(
                                f"Ollama returned relevance for an unexpected crop '{crop_name}'."
                            )

                    log.info("Successfully parsed Ollama news context JSON analysis.")
                    return impact_summary, news_mapping
                else:
                    log.error(
                        f"Ollama JSON output was not a dictionary: {type(llm_result)}"
                    )

            except json.JSONDecodeError as json_e:
                log.error(f"Failed to decode JSON from Ollama response: {json_e}")
                log.error(f"Ollama raw content was: {raw_content}")
        else:
            log.error("Ollama response structure was unexpected or missing content.")

    except ollama.ResponseError as e:
        log.error(
            f"Ollama API Response Error (News Analysis): {e.status_code} - {e.error}"
        )
        if e.status_code == 404:
            log.error(
                f"Model '{OLLAMA_MODEL}' not found at {OLLAMA_BASE_URL}. Pull it with `ollama pull {OLLAMA_MODEL}`."
            )
        return f"Ollama API Error: {e.error}", news_mapping
    except Exception as e:
        log.error(
            f"Error invoking direct Ollama call (News Analysis): {e}", exc_info=True
        )
        return f"Error during Ollama call: {e}", news_mapping

    # Return defaults if parsing or call failed
    return default_impact_summary, news_mapping


# --- Other functions (generate_explanation, get_primary_risks) - No LLM calls ---
def generate_explanation(
    crop_name: str, score: float, factors: Dict, context: Dict, status_info: Dict
) -> List[ExplanationPoint]:
    """Generates explanation points based on scoring factors and context."""
    points = []
    pf = factors.get("profitability", {})
    sf = factors.get("subsidy", {})
    rf = factors.get("risk", {})

    soil_type = context.get("soil_type", "provided")
    profit_score_val = pf.get("score", 1.0)
    profit_qualifier = (
        "strong"
        if profit_score_val > 1.1
        else "moderate"
        if profit_score_val > 0.9
        else "lower"
    )
    profit_detail = f"Estimated profitability is {profit_qualifier}, driven by price trend and relative input costs ({pf.get('details', 'N/A')})."
    points.append(
        ExplanationPoint(
            reason_type="Profitability Analysis",
            detail=profit_detail,
            ai_component="Predictive Analytics (Price), Heuristic Evaluation (Yield/Cost)",
        )
    )

    agronomic_info: Optional[AgronomicInfo] = data_sources.get_crop_agronomic_data(
        crop_name
    )
    water_needs = agronomic_info.water_needs if agronomic_info else "Unknown"
    points.append(
        ExplanationPoint(
            reason_type="Agronomic Suitability",
            detail=f"Good match for {soil_type} soil. Water needs ({water_needs}) considered against forecast.",
            ai_component="Knowledge-Based Rules (Soil), Weather Data Integration",
        )
    )

    subsidy_score_val = sf.get("score", 1.0)
    if subsidy_score_val > 1.0:
        subsidy_detail = f"Financial viability potentially enhanced by {sf.get('details', 'available support schemes')}."
        component = "Rule-Based Logic (Subsidy Impact)"
        if status_info.get("status") == "high_priority_subsidy":
            subsidy_detail = f"Financial viability significantly boosted by high-impact subsidy: {sf.get('details', 'available support')}. Game-changer identified."
            component = "Rule-Based Logic (High-Impact Subsidy Detection)"
        points.append(
            ExplanationPoint(
                reason_type="Financial Incentive (Subsidy)",
                detail=subsidy_detail,
                ai_component=component,
            )
        )

    risk_detail = f"Calculated risk profile considers {rf.get('details', 'standard factors including price volatility and news sentiment')}. See primary risks below."
    points.append(
        ExplanationPoint(
            reason_type="Risk Assessment",
            detail=risk_detail,
            ai_component="Risk Factor Analysis (Volatility, News Sentiment)",
        )
    )

    points.append(
        ExplanationPoint(
            reason_type="Methodology Note",
            detail="Recommendation derived using a hybrid approach combining rules, predictive signals, scoring models, and a knowledge base.",
            ai_component="Hybrid AI System",
        )
    )

    log.debug(
        f"[{crop_name}] Generated explanation points: {[p.reason_type for p in points]}"
    )
    return points


def get_primary_risks(
    crop_name: str, factors: Dict, events: List[Dict], region_code: str
) -> List[str]:
    """Identifies primary risks based on factors, agronomic data, and critical events."""
    risks = set()
    rf = factors.get("risk", {})
    agronomic_info: Optional[AgronomicInfo] = data_sources.get_crop_agronomic_data(
        crop_name
    )

    # 1. Risk from scoring factors (Volatility)
    risk_details_lower = rf.get("details", "").lower()
    if "high volatility" in risk_details_lower:
        risks.add("Potential high price volatility.")
    elif "medium volatility" in risk_details_lower:
        risks.add("Moderate price volatility expected.")
    elif "low volatility" in risk_details_lower:
        risks.add("Price volatility expected to be relatively low.")

    # 2. Risks from Agronomic Data
    if agronomic_info:
        if agronomic_info.common_pests:
            risks.add(
                f"Monitor common pests/diseases: {', '.join(agronomic_info.common_pests)}."
            )
        if (
            agronomic_info.general_notes
            and "risk" in agronomic_info.general_notes.lower()
        ):
            risks.add(f"Agronomic Note: {agronomic_info.general_notes}")
        if "Soybean" == crop_name:
            risks.add("Sensitivity to waterlogging, ensure good drainage.")
        if "Maize" in crop_name:
            risks.add("Yield sensitive to dry spells during tasseling/silking.")
        if "Cotton" == crop_name:
            risks.add(
                "High input requirements & significant pest pressure (esp. Bollworm)."
            )
        if "Paddy" in crop_name:
            risks.add("Requires significant water; dependent on monsoon/irrigation.")
        if "Sugarcane" == crop_name:
            risks.add("Long duration crop; high water/nutrient needs.")
        if "Chickpea" in crop_name:
            risks.add("Susceptible to wilt diseases in some soils.")
        if "Tomato" in crop_name:
            risks.add("Prone to various viral diseases (e.g., Leaf Curl).")

    # 3. Risks from Critical Events
    for event in events:
        event_regions = event.get("region_codes", [])
        applies_to_region = not event_regions or region_code in event_regions
        if applies_to_region and crop_name in event.get("crop_affected", []):
            severity = event.get("severity", "medium").lower()
            summary = event.get("summary", "Unspecified Event")
            event_type = event.get("type", "Alert")
            risks.add(f"EVENT ({severity.upper()}): {event_type} - {summary}")

    final_risks = sorted(list(risks))
    log.debug(f"[{crop_name}] Identified risks: {final_risks}")
    return final_risks if final_risks else ["Standard agricultural risks apply."]


# --- Generate Final Summary (Using direct ollama call) ---
def generate_llm_summary(
    recommendations: List[Recommendation],
    news_impact_summary: str,
    weather_summary: str,
) -> Optional[str]:
    """Uses direct ollama library call to generate an overall summary paragraph."""
    log.info("Generating final summary with direct Ollama call...")
    if not recommendations:
        log.warning("No recommendations available to generate summary.")
        return "No suitable crop recommendations were generated based on the input criteria."

    # Construct prompt manually
    system_prompt = (
        "You are an expert agricultural advisor AI. Your goal is to provide a clear, concise, and actionable summary based on the provided analysis context and top crop recommendations. "
        "Address the user directly (e.g., 'Based on your inputs...'). "
        "Briefly mention the weather and news context. "
        "Highlight the top 1-2 recommended crops, mentioning their key strengths (like profitability potential or suitability) and perhaps a major risk to watch for. "
        "Keep the tone encouraging but realistic. Aim for a short paragraph (3-5 sentences)."
        "Do not just list the recommendations again; synthesize the information."
    )

    recs_formatted = ""
    for rec in recommendations[:2]:  # Limit context to top 2
        profit_text = next(
            (
                p.detail.split(" is ")[-1].split(",")[0]
                for p in rec.explanation_points
                if p.reason_type == "Profitability Analysis"
            ),
            "N/A",
        )
        subsidy_text = next(
            (p.detail for p in rec.explanation_points if "Subsidy" in p.reason_type),
            None,
        )
        risk_text = (
            rec.primary_risks[0].split(":")[0].replace(".", "")
            if rec.primary_risks
            else "standard risks"
        )
        recs_formatted += f"\n- Rank {rec.rank}: **{rec.crop_name}** (Score: {rec.recommendation_score:.1f})"
        recs_formatted += f"\n  - Key Point: Estimated {profit_text} profitability."
        if subsidy_text and "significantly boosted" in subsidy_text.lower():
            recs_formatted += f"\n  - Note: Benefits from significant subsidy support."

    user_prompt = (
        f"Analysis Context:\n"
        f"- Location/Soil: Provided by user.\n"
        f"- Weather Summary: {weather_summary}\n"
        f"- Recent News Context Summary: {news_impact_summary}\n\n"
        f"Top Recommendations (Details provided separately):\n"
        f"{recs_formatted}\n\n"
        "Task: Generate a user-friendly summary paragraph based on the context and the key aspects of the top recommendations. Focus on the overall picture for the user.\n"
        "Summary Paragraph:"
    )

    try:
        log.info("Invoking ollama.chat for final summary...")
        # Use the ollama client configured with the base URL from config
        client = ollama.Client(host=OLLAMA_BASE_URL)  # Initialize client with host URL
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # No format='json' needed here, expecting text
            options={"temperature": LLM_TEMPERATURE},
        )

        if response and "message" in response and "content" in response["message"]:
            summary = response["message"]["content"].strip()
            log.info("Ollama generated final summary successfully.")
            return summary
        else:
            log.error(
                "Ollama summary response structure was unexpected or missing content."
            )
            return "Summary generation failed (unexpected response)."

    except ollama.ResponseError as e:
        log.error(f"Ollama API Response Error (Summary): {e.status_code} - {e.error}")
        if e.status_code == 404:
            log.error(
                f"Model '{OLLAMA_MODEL}' not found at {OLLAMA_BASE_URL}. Pull it with `ollama pull {OLLAMA_MODEL}`."
            )
        return f"Summary generation failed (Ollama API Error: {e.error})."
    except Exception as e:
        log.error(f"Error invoking direct Ollama call (Summary): {e}", exc_info=True)
        return f"Summary generation failed (Error: {e})."


import time

SPECIAL_KEYS_NO_TRANSLATE = {
    "$id",
    "$createdAt",
    "$updatedAt",
    "$permissions",
    "$databaseId",
    "$collectionId",
    "provider",
}

MAX_BATCH_CHARS = 1500
MAX_BATCH_ITEMS = 50

# Date regex to detect date-like strings
_date_regex = re.compile(
    r"""^(
        \d{4}-\d{2}-\d{2} |         # 2025-05-06
        \d{2}/\d{2}/\d{4} |         # 06/05/2025
        \d{4}/\d{2}/\d{2} |         # 2025/05/06
        \d{2}-\d{2}-\d{4} |         # 06-05-2025
        \d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*[+-]\d{2}:\d{2} # 06-05-2025, 19:26:07.533 +00:00
    )$""",
    re.VERBOSE,
)


def _is_date_string(s: str) -> bool:
    """Check if a string is a date or datetime format."""
    return bool(_date_regex.match(s.strip()))


def _extract_json_content(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` from a wrapped string."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def batch_texts(texts: list[str]) -> Generator[list[str], None, None]:
    """Split a list of texts into batches that respect max character and item limits."""
    batch = []
    char_count = 0

    for text in texts:
        if (len(batch) >= MAX_BATCH_ITEMS and False) or char_count + len(
            text
        ) > MAX_BATCH_CHARS:
            yield batch
            batch = []
            char_count = 0
        batch.append(text)
        char_count += len(text)

    if batch:
        yield batch


def translate_texts(texts: list[str], target_lang: str) -> list[str]:
    """Translate a list of texts using Ollama in batches."""
    all_translations = []

    for i, batch in enumerate(batch_texts(texts)):
        prompt = (
            f"Translate the following {len(batch)} items to {target_lang} (use native script). "
            f"Do NOT translate dates, times, or links and if possible use only native script characters, not original language characters. Only output a JSON array of translated texts, in order. "
            f"Strictly return valid JSON.\n\n{json.dumps(batch, ensure_ascii=False)}"
        )

        log.info(f"Translating batch {i + 1} of size {len(batch)}")
        start_time = time.time()

        try:
            client = ollama.Client(host=OLLAMA_BASE_URL)
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful translation assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": LLM_TEMPERATURE},
            )

            elapsed = time.time() - start_time
            log.info(f"Batch {i + 1} translated in {elapsed:.2f} seconds")

            raw_content = response.get("message", {}).get("content", "")
            extracted = _extract_json_content(raw_content.strip())
            translated = json.loads(extracted)

            if isinstance(translated, list):
                all_translations.extend(translated)
            else:
                raise ValueError("Expected list in translation response.")

        except Exception as e:
            log.error(f"Translation failed in {time.time() - start_time:.2f} sec: {e}")
            all_translations.extend(batch)

    return all_translations


def translate_json_values(data: Any, target_lang: str) -> Any:
    """
    Traverse a nested JSON-like structure, collect translatable values,
    translate them, and re-apply them in the same structure.
    """
    translations = []
    paths = []

    def collect(d, path=(), in_dynamic=False):
        if isinstance(d, str) and not _is_date_string(d):
            translations.append(d)
            paths.append(path)
        elif isinstance(d, dict):
            is_dynamic = in_dynamic or (path and path[-1] == "dynamic_fields")
            for k, v in d.items():
                key_path = path + (k,)
                if is_dynamic and not _is_date_string(k):
                    translations.append(k)
                    paths.append(path + ("__key__",))
                if isinstance(v, (dict, list)) or (
                    isinstance(v, str)
                    and not _is_date_string(v)
                    and k not in SPECIAL_KEYS_NO_TRANSLATE
                    and "link" not in k
                ):
                    collect(v, key_path, is_dynamic)
        elif isinstance(d, list):
            for idx, item in enumerate(d):
                collect(item, path + (idx,), in_dynamic)

    collect(data)

    translated_texts = translate_texts(translations, target_lang)
    path_map = {p: t for p, t in zip(paths, translated_texts)}

    def apply(d, path=()):
        if isinstance(d, dict):
            result = {}
            for k, v in d.items():
                key_path = path + ("__key__",)
                val_path = path + (k,)
                new_k = path_map.get(key_path, k)
                result[new_k] = apply(v, val_path)
            return result
        elif isinstance(d, list):
            return [apply(item, path + (i,)) for i, item in enumerate(d)]
        elif isinstance(d, str):
            return path_map.get(path, d)
        else:
            return d

    return apply(data)


def _translate_text_with_ollama(text: str, target_lang: str) -> str:
    prompt = (
        f"Translate the following text to {target_lang} and do not translate dates, time, and links. "
        f"Only return translated text in native script, if possible use only native script characters, not original language characters and give no explanations:\n\n{text}"
    )
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful translation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": LLM_TEMPERATURE},
        )
        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"].strip() or text
        else:
            log.error("Ollama translation response missing content.")
            return text
    except Exception as e:
        log.error(f"Ollama translation failed: {e}")
        return text


def _translate_name_key(data: Any, target_lang: str) -> Any:
    """
    Recursively translates only the 'name' key in the given dict to the target language.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k == "name" and isinstance(v, str):
                result[k] = _translate_text_with_ollama(v, target_lang)
            else:
                result[k] = _translate_name_key(v, target_lang)
        return result
    elif isinstance(data, list):
        return [_translate_name_key(item, target_lang) for item in data]
    else:
        return data


def get_translations(
    input_json: Union[Dict[str, Any], str], language: str = "english"
) -> Any:
    """
    Translates the input JSON (English) to the specified language.
    For 'english', only the 'name' key is translated using LLM.
    For 'hindi' and 'marathi', all values are translated.
    Args:
        input_json: dict or JSON string.
        language: 'hindi', 'marathi', or 'english'.
    Returns:
        The translated JSON (or original if language is not supported).
    """
    log.info(f"--- Received request for translation to '{language}' ---")
    try:
        if isinstance(input_json, str):
            data = json.loads(input_json)
        elif isinstance(input_json, dict):
            data = input_json
        else:
            log.error(f"Invalid input type: {type(input_json)}")
            return input_json  # fallback: return as-is

        if language == "english":
            # Only translate the 'name' key using LLM
            return _translate_name_key(data, "english")
        elif language in ["marathi", "hindi"]:
            return translate_json_values(data, language)
        else:
            log.error("language must be one of: 'hindi', 'marathi', 'english'.")
            return data
    except Exception as e:
        log.error(f"Translation failed: {e}", exc_info=True)
        return input_json  # fallback: return as-is


def translate_string(text: str, possible_outputs: list[str]) -> str:
    """
    Uses Ollama LLM to translate the input string to English, but ONLY to one of the allowed outputs.
    If not possible, returns 'Error: Cannot be translated'.
    """
    prompt = (
        "You are a translation assistant. "
        "Translate the following input to English, but ONLY use one of these allowed outputs: "
        f"{possible_outputs}. "
        "If the input cannot be translated to any of these, return the input string exactly as it is.\n\n"
        f"Input: {text}\n"
        "Output:"
    )
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful translation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": LLM_TEMPERATURE},
        )
        if response and "message" in response and "content" in response["message"]:
            result = response["message"]["content"].strip()
            if result in possible_outputs:
                return result
            else:
                return "Error: Cannot be translated"
        else:
            log.error("Ollama translation-to-allowed response missing content.")
            return "Error: Cannot be translated"
    except Exception as e:
        log.error(f"Ollama translation-to-allowed failed: {e}")
        return "Error: Cannot be translated"
