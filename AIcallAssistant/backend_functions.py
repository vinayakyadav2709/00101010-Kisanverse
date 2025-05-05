# --- START OF FILE backend_functions.py ---
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import os

logger = logging.getLogger(__name__)
# ... (Load Static Data & _load_json_data) ...
ALL_DATA_PATH = "all_data.json"; RECOMMENDATION_DATA_PATH = "recommendation_output.json"; SS_DATA_PATH = "ss.json"; WEATHER_LIST_PATH = "weather_list.json"
def _load_json_data(file_path: str) -> Optional[Dict]:
    if not os.path.exists(file_path): logger.error(f"Data file missing: {file_path}"); return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        logger.info(f"Loaded data: {file_path}"); return data
    except Exception as e: logger.error(f"Error loading {file_path}: {e}", exc_info=True); return None
all_data = _load_json_data(ALL_DATA_PATH); recommendation_data = _load_json_data(RECOMMENDATION_DATA_PATH)
ss_data = _load_json_data(SS_DATA_PATH); weather_list_data = _load_json_data(WEATHER_LIST_PATH)

def _get_default_location(kwargs: Dict[str, Any], default: str = "default_area") -> str:
    loc = kwargs.get('location'); return loc.strip().lower() if loc and isinstance(loc, str) and loc.strip() else default

# --- Tool Functions ---
def add_crop_listing(**kwargs: Any) -> Dict[str, Any]:
    farmer = kwargs.get('farmer_id', 'caller'); crop = str(kwargs.get('crop_type', '')).upper().strip(); price_str = str(kwargs.get('price_per_kg', '0')); quantity_str = str(kwargs.get('total_quantity', '0')); location = _get_default_location(kwargs)
    logger.info(f"--- Tool: add_crop_listing --- Args: farmer={farmer}, crop='{crop}', price_str='{price_str}', qty_str='{quantity_str}', loc={location}")
    errors = []; price = 0; quantity = 0
    if not crop: errors.append("Crop type missing.")
    try: price = int(price_str); assert price > 0
    except: errors.append(f"Invalid price: '{price_str}'.")
    try: quantity = int(quantity_str); assert quantity > 0
    except: errors.append(f"Invalid quantity: '{quantity_str}'.")
    if errors: error_message = "Cannot add listing: " + " ".join(errors); logger.warning(error_message); return {"success": False, "error": error_message, "listing_id": None}
    simulated_id = f"NEWLIST-{random.randint(1000,9999)}"; logger.info(f"Simulated adding listing ID: {simulated_id}")
    success_message = f"आपकी {quantity} किलो {crop} की लिस्टिंग {price} रुपये प्रति किलो के हिसाब से बना दी गई है।" # Hindi success message
    return {"success": True, "message": success_message, "listing_id": simulated_id}

# --- MODIFIED: get_marketplace_orders ---
def get_marketplace_orders(**kwargs: Any) -> Dict[str, Any]:
    product_filter = kwargs.get('product_name', '').upper()
    listing_id_filter = kwargs.get('listing_id')
    location = _get_default_location(kwargs)
    logger.info(f"--- Tool: get_marketplace_orders --- Args: product={product_filter}, listing_id={listing_id_filter}, loc={location}")

    # --- ADDED: Demo Scenario Handling ---
    # If checking status for a simulated listing ID, return success directly
    if listing_id_filter and listing_id_filter.startswith("NEWLIST-"):
        logger.info(f"Demo Scenario: Handling status check for simulated listing ID {listing_id_filter}")
        status_summary = "आपकी लिस्टिंग पूरी हो गई है।" # Your listing is completed.
        payment_status = "हाँ, भुगतान आपके लिंक्ड खाते में सफलतापूर्वक प्राप्त हो गया है (सिमुलेटेड)।" # Yes, payment successfully received in linked account (simulated).
        return {
            "success": True, # Indicate success to app.py
            "orders_found": 1,
            "details": [{
                "listing_id": listing_id_filter,
                "crop_type": "WHEAT", # Assuming wheat from demo
                "status_summary": status_summary,
                "payment_status_simulated": payment_status
            }]
        }
    # --- END ADDED ---

    # Original logic for non-demo listing ID checks or general searches
    if not all_data or "listings" not in all_data:
        return {"success": False, "error": "Marketplace listing data not available."} # Use error key

    listings = all_data["listings"]
    results = []

    if listing_id_filter:
        listing = listings.get(listing_id_filter)
        if listing:
            details = listing.get("details", {}); bids = listing.get("bids", [])
            fulfilled_qty = sum(b.get('quantity', 0) for b in bids if b.get('status') == 'fulfilled'); accepted_qty = sum(b.get('quantity', 0) for b in bids if b.get('status') == 'accepted')
            pending_bids = [b for b in bids if b.get('status') == 'pending']; status = details.get("status", "unknown"); available = details.get("available_quantity", 0)
            payment_status = "भुगतान की स्थिति लंबित है।"
            if status == 'listed' and (fulfilled_qty + accepted_qty) >= details.get('total_quantity', 0): status_summary = "आपकी लिस्टिंग पूरी हो गई है या पूरी तरह से स्वीकार कर ली गई है।"; payment_status = "भुगतान आपके लिंक्ड खाते में संसाधित कर दिया गया है (सिमुलेटेड)।"
            elif pending_bids: status_summary = f"लिस्टिंग सक्रिय है। उपलब्ध: {available} किलो। {len(pending_bids)} बोलियां लंबित हैं।"
            elif available > 0 : status_summary = f"लिस्टिंग सक्रिय है। उपलब्ध: {available} किलो। कोई लंबित बोली नहीं।"
            else: status_summary = f"लिस्टिंग स्थिति: {status}। उपलब्ध: {available} किलो।"
            results.append({"listing_id": listing_id_filter, "crop_type": details.get("crop_type"), "original_quantity": details.get("total_quantity"), "available_quantity": available, "status_summary": status_summary, "payment_status_simulated": payment_status})
        else:
            # Return consistent error structure
            return {"success": False, "error": f"Listing ID {listing_id_filter} not found in static data."}
    else: # General search removed for demo simplicity
         return {"success": False, "message": "Please provide a specific listing ID to check status."} # Use message key

    return {"success": True, "orders_found": len(results), "details": results[:5]}

# ... (get_weather_forecast, get_crop_recommendations, get_subsidy_info, get_contracts remain the same) ...
def get_weather_forecast(**kwargs: Any) -> Dict[str, Any]:
    location = _get_default_location(kwargs); logger.info(f"--- Tool: get_weather_forecast --- Args: loc={location}")
    if not weather_list_data or not isinstance(weather_list_data, list): return {"error": "Weather forecast data not available."}
    forecast_data = weather_list_data[:7]; result = {"location_context": location, "data_source": "weather_list.json (Static)", "forecast_period": f"{len(forecast_data)} days", "forecast": forecast_data}
    logger.info(f"--- Result (get_weather_forecast): Returned {len(forecast_data)} days forecast ---"); return result
def get_crop_recommendations(**kwargs: Any) -> Dict[str, Any]:
    location = _get_default_location(kwargs); logger.info(f"--- Tool: get_crop_recommendations --- Args: loc={location}")
    if not recommendation_data or "recommendations" not in recommendation_data: return {"error": "Crop recommendation data not available."}
    recommendations = recommendation_data.get("recommendations", []); summary = recommendation_data.get("overall_llm_summary", "No summary."); request_details = recommendation_data.get("request_details", {})
    top_recs = [{"rank": rec.get("rank"), "crop_name": rec.get("crop_name"), "score": rec.get("recommendation_score"), "yield_range": rec.get("key_metrics", {}).get("expected_yield_range"), "price_trend": rec.get("key_metrics", {}).get("price_forecast_trend")} for rec in recommendations[:2]]
    result = {"location_context": f"{request_details.get('latitude', 'N/A')}, {request_details.get('longitude', 'N/A')} (Soil: {request_details.get('soil_type', 'N/A')})", "top_recommendations": top_recs, "overall_summary": summary, "data_source": "recommendation_output.json (Static)"}
    logger.info(f"--- Result (get_crop_recommendations): Returned {len(top_recs)} recommendations ---"); return result
def get_subsidy_info(**kwargs: Any) -> Dict[str, Any]:
    scheme_filter = kwargs.get('scheme_name'); location = _get_default_location(kwargs); logger.info(f"--- Tool: get_subsidy_info --- Args: scheme={scheme_filter}, loc={location}")
    available_subsidies = []
    if all_data and "subsidies" in all_data:
        all_subs = all_data["subsidies"]
        for sid, sub_data in all_subs.items():
            details = sub_data.get("details"); 
            if not details: continue
            subsidy_locations = [loc.upper() for loc in details.get("locations", [])]
            loc_match = (location == 'default_area' or not subsidy_locations or any(location.upper() in sub_loc for sub_loc in subsidy_locations))
            if not loc_match: continue
            scheme_match = (not scheme_filter or (scheme_filter.lower() in details.get("program", "").lower() or scheme_filter.lower() in details.get("provider", "").lower()))
            if not scheme_match: continue
            dynamic_data = {}; 
            try: dynamic_data = json.loads(details.get("dynamic_fields","{}")) 
            except: pass
            available_subsidies.append({"program_name": details.get("program", "N/A"), "provider": details.get("provider", "N/A"), "description": details.get("description"), "benefits": details.get("benefits"), "eligibility": details.get("eligibility"), "application_link": dynamic_data.get("application_link")})
    if location in ['mumbai', 'pune', 'maharashtra']:
        logger.info(f"Location '{location}' triggered Maharashtra subsidy simulation.")
        if len(available_subsidies) < 2: available_subsidies.append({"program_name": "महाबीज बियाणे अनुदान योजना (सिम्युलेटेड)", "provider": "महाराष्ट्र शासन कृषी विभाग", "description": "उन्नत आणि प्रमाणित बियाणे खरेदीसाठी अनुदान.", "benefits": "बियाणे खर्चावर ५०% पर्यंत अनुदान.", "eligibility": "नोंदणीकृत शेतकरी, विशेषतः अल्पभूधारक.", "application_link": "महाडीबीटी पोर्टलवर अर्ज करा (सिम्युलेटेड लिंक)"})
        if len(available_subsidies) < 2: available_subsidies.append({"program_name": "मुख्यमंत्री सिंचन योजना (सिम्युलेटेड)", "provider": "महाराष्ट्र शासन जलसंधारण विभाग", "description": "ठिबक किंवा तुषार सिंचन संच बसवण्यासाठी आर्थिक सहाय्य.", "benefits": "खर्चाच्या ७५% पर्यंत अनुदान (पात्रतेनुसार).", "eligibility": "सर्व शेतकरी, कोरडवाहू क्षेत्रांना प्राधान्य.", "application_link": "संबंधित कृषी कार्यालयात संपर्क साधा (सिम्युलेटेड लिंक)"})
    num_found = len(available_subsidies); results_to_return = available_subsidies[:3]
    if not results_to_return:
        msg = "तुमच्या निकषांशी जुळणारी कोणतीही विशिष्ट सबसिडी सध्या आढळली नाही." if location in ['mumbai', 'pune', 'maharashtra'] else "No specific subsidies found matching your criteria."
        return {"success": False, "message": msg} # Indicate failure but with a user message
    result = {"success": True, "query": {"scheme_filter": scheme_filter, "location_context": location}, "subsidies_found": num_found, "details": results_to_return}
    logger.info(f"--- Result (get_subsidy_info): Found {num_found}, returning {len(results_to_return)} ---"); return result
def get_contracts(**kwargs: Any) -> Dict[str, Any]:
    crop_filter = kwargs.get('crop_name', '').upper(); company_filter = kwargs.get('company', '').upper(); location = _get_default_location(kwargs)
    logger.info(f"--- Tool: get_contracts --- Args: crop={crop_filter}, company={company_filter}, loc={location}")
    if not all_data or "contracts" not in all_data: return {"error": "Contract data not available."}
    all_contracts_dict = all_data["contracts"]; matching_contracts = []
    for cid, contract_data in all_contracts_dict.items():
        details = contract_data.get("details"); 
        if not details: continue
        contract_locations = [loc.upper() for loc in details.get("locations", [])]
        loc_match = (location == 'default_area' or not contract_locations or any(location.upper() in cl for cl in contract_locations))
        crop_match = (not crop_filter or details.get("crop_type", "").upper() == crop_filter)
        company_match = (not company_filter) # Simplified
        if loc_match and crop_match and company_match: matching_contracts.append({"contract_id": cid, "crop_type": details.get("crop_type"), "quantity_kg": details.get("quantity"), "price_per_kg": details.get("price_per_kg"), "status": details.get("status"), "location_info": ", ".join(details.get("locations",[]))})
    if not matching_contracts: return {"message": "No active contracts found matching your criteria."}
    return {"query": {"crop": crop_filter, "company": company_filter, "location": location}, "contracts_found": len(matching_contracts), "details": matching_contracts[:5]}

# --- END OF FILE backend_functions.py ---