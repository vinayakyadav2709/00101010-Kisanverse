# --- START OF FILE app.py ---
import logging
from flask import Flask, request, Response
# Remove json, pydantic imports if only using hardcoded responses
# import json
# from pydantic import ValidationError
from typing import Optional, Dict, Any, List
import time
import os
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
import config
import llm_handler # Still needed for non-demo flows if any
import conversation_manager as cm
import twilio_handler as th
from stt_selector import get_stt_client
# Removed DummyBackend and bf import as we are hardcoding for demo
from models import STTResult # Keep STTResult

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Initialize clients
twilio_client: Optional[TwilioClient] = None
if config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN:
    try: twilio_client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN); logger.info("Twilio client OK.")
    except Exception as e: logger.error(f"Twilio client init fail: {e}", exc_info=True)
else: logger.warning("Twilio credentials missing.")
stt_client = get_stt_client()
if not stt_client: logger.critical("STT client FAILED init.")
llm_client_present = bool(llm_handler.llm_client) # Check if LLM client loaded
if not llm_client_present: logger.warning("LLM client NOT loaded. Only hardcoded demo flows will work.")

# Constants & Helpers
REQUESTED_LANGUAGES = {'en', 'hi', 'mr'} # Limit to demo languages
RELEVANT_LANGUAGES = {info['lang_code'] for _, info in config.LOCATION_TO_LANGUAGE.items() if info['lang_code'] in REQUESTED_LANGUAGES}; RELEVANT_LANGUAGES.add('en')
SUPPORTED_LANGUAGE_NAMES = [info['lang_name'] for _, info in config.LOCATION_TO_LANGUAGE.items() if info['lang_code'] in RELEVANT_LANGUAGES]
unique_lang_names = sorted(list(set(SUPPORTED_LANGUAGE_NAMES)));
if "Hindi" in unique_lang_names: unique_lang_names.insert(0, unique_lang_names.pop(unique_lang_names.index("Hindi")))
if "English" in unique_lang_names: unique_lang_names.insert(0, unique_lang_names.pop(unique_lang_names.index("English")))
SUPPORTED_LANGUAGES_STR = ", ".join(unique_lang_names); logger.info(f"Relevant languages: {RELEVANT_LANGUAGES}")
def is_relevant_language(lang_code: Optional[str]) -> bool: return lang_code in RELEVANT_LANGUAGES if lang_code else False
call_context: Dict[str, Dict[str, Any]] = {}; 
def get_call_context(call_sid: str) -> Dict[str, Any]: return call_context.setdefault(call_sid, {})
def clear_call_context(call_sid: str):
    if call_sid in call_context: del call_context[call_sid]
def log_conversation_turn(role: str, text: str, lang: Optional[str] = None, call_sid: Optional[str] = None):
    sid_info = f"[SID:{call_sid[-4:]}]" if call_sid else ""; lang_info = f" ({lang})" if lang else ""; loc = get_call_context(call_sid).get('location', ''); loc_info = f" (Loc:{loc})" if loc else ""
    print(f"\n{'='*5} {role.upper()} {sid_info}{loc_info}{lang_info} {'='*5}"); print(f"{text}"); print(f"{'='*(12+len(role)+len(sid_info)+len(loc_info)+len(lang_info))}\n")
    log_text = text[:500] + ('...' if len(text) > 500 else ''); logger.info(f"TURN {sid_info}{loc_info}: Role={role}, Lang={lang}, Text='{log_text}'")

# --- Routes ---
@app.route("/voice", methods=['POST'])
def handle_incoming_call():
    call_sid = request.values.get('CallSid'); location_raw = request.values.get('Location', 'unknown')
    location = location_raw.lower().strip() if location_raw else 'unknown'; logger.info(f"--- Incoming Call --- SID: {call_sid}, Loc: {location}")
    if not call_sid: return Response(th.create_error_response(config.ERROR_MESSAGE), mimetype='text/xml')
    lang_info = config.LOCATION_TO_LANGUAGE.get(location, config.LOCATION_TO_LANGUAGE.get('unknown', config.LOCATION_TO_LANGUAGE['default']))
    code, name, twilio_code = lang_info['lang_code'], lang_info['lang_name'], lang_info['twilio_code']
    if code not in RELEVANT_LANGUAGES and code != 'en': logger.warning(f"Loc '{location}' non-relevant lang '{code}'. Fallback EN."); code, name, twilio_code = 'en', 'English', 'en-IN'
    context = get_call_context(call_sid); context.update({'location': location, 'language_code': code, 'language_name': name, 'twilio_code': twilio_code, 'demo_scenario': None, 'last_listing_id': '1009'}) # Pre-set listing ID for demo
    logger.info(f"Call Context Init: {context}")
    cm.clear_conversation_history(call_sid); system_prompt = llm_handler.get_system_prompt(location=location, language=f"{name} ({code})"); cm.initialize_history(call_sid, system_prompt['content'])
    greeting_template = config.GREETING_TEMPLATES.get(code, config.GREETING_TEMPLATES['en']); initial_greeting = greeting_template.format(location=location.capitalize(), language=name)
    log_conversation_turn("SYSTEM", f"Call initiated. Lang: {code}. Greeting: {initial_greeting}", call_sid=call_sid)
    twiml = th.create_greeting_gather_response(initial_greeting, '/process-speech', twilio_code); return Response(twiml, mimetype='text/xml')

def get_latest_recording_url(call_sid: str, max_retries: int = 3, delay: int = 2) -> Optional[str]:
    if not twilio_client: return None
    for attempt in range(max_retries):
        logger.info(f"API Fetch Rec Attempt {attempt + 1}/{max_retries} for {call_sid[-4:]}...")
        try:
            recs = twilio_client.recordings.list(call_sid=call_sid, limit=1)
            if recs: rec = recs[0]; uri = rec.uri.replace('.json', '.mp3'); uri = f"/2010-04-01{uri}" if not uri.startswith('/2010-04-01/') else uri; url = f"https://api.twilio.com{uri}"; logger.info(f"API Found: {rec.sid}, St={rec.status}"); return url if rec.status == 'completed' else logger.warning(f"Status '{rec.status}'. Wait...") or None
            else: logger.warning(f"No recordings API yet.")
        except TwilioRestException as e: logger.error(f"Twilio API Err: {e}"); return None
        except Exception as e: logger.error(f"API Err: {e}", exc_info=True); return None
        if attempt < max_retries - 1: time.sleep(delay)
    logger.error(f"Failed API fetch for {call_sid[-4:]}."); return None

@app.route("/process-speech", methods=['POST'])
def process_speech_input():
    call_sid, _, recording_url_hook = th.extract_call_data(request.values)
    if not call_sid: return Response(th.create_error_response(config.ERROR_MESSAGE), mimetype='text/xml')

    logger.info(f"--- Processing Speech --- SID: {call_sid[-4:]}")
    context = get_call_context(call_sid)
    if not context: loc, lang_code, tw_code = "unknown", 'en', 'en-IN'; logger.error(f"Context LOST for {call_sid[-4:]}!")
    else: loc, lang_code, tw_code = context.get('location', 'unk'), context.get('language_code', 'en'), context.get('twilio_code', 'en-IN')
    history = cm.get_conversation_history(call_sid) # Keep history for potential non-demo flows
    if not history: logger.error(f"History LOST {call_sid[-4:]}."); return Response(th.create_error_response(config.CONVERSATION_ERROR_MESSAGE, response_language=tw_code), mimetype='text/xml')
    rec_url = recording_url_hook or get_latest_recording_url(call_sid)
    if not rec_url: logger.error(f"Failed get RecordingUrl {call_sid[-4:]}."); return Response(th.create_error_response("Cannot retrieve recording.", response_language=tw_code), mimetype='text/xml')
    if not stt_client: logger.error(f"STT Client unavailable."); return Response(th.create_error_response(config.ERROR_MESSAGE, response_language=tw_code), mimetype='text/xml')
    logger.info(f"Calling STT ({config.STT_PROVIDER}), Lang Hint: {lang_code}")
    stt_dict = stt_client.transcribe_audio(rec_url, expected_language=lang_code) # Force language

    if stt_dict is None: logger.error(f"STT failed {call_sid[-4:]}."); return Response(th.create_error_response(config.ERROR_MESSAGE, response_language=tw_code), mimetype='text/xml')
    try: stt_res = STTResult(**stt_dict); logger.info(f"STT OK: Rel={stt_res.is_reliable}, Lang={stt_res.language}, Text='{stt_res.text[:50]}...'")
    except ValidationError as e: logger.error(f"STT validation failed: {e}"); return Response(th.create_error_response(config.ERROR_MESSAGE, response_language=tw_code), mimetype='text/xml')

    if not stt_res.is_reliable or not stt_res.text.strip():
        logger.warning(f"STT unreliable/empty {call_sid[-4:]}. Asking repeat."); log_conversation_turn("SYSTEM", f"STT unreliable/empty.", call_sid=call_sid)
        twiml = th.create_no_speech_response(config.NO_SPEECH_MESSAGE, '/process-speech', config.LISTEN_PROMPT, tw_code); return Response(twiml, mimetype='text/xml')

    # --- Hardcoded Demo Logic ---
    user_text_lower = stt_res.text.lower()
    final_spoken_response = None
    response_lang_twilio = tw_code # Default to call context language
    hangup_call = False

    # Marketplace Demo (Hindi) - Check if preferred lang is Hindi
    if lang_code == 'hi':
        if any(kw in user_text_lower for kw in config.DEMO_RESPONSES["marketplace"]["triggers_sell"]):
            final_spoken_response = config.DEMO_RESPONSES["marketplace"]["response_list_success"]
            response_lang_twilio = config.DEMO_RESPONSES["marketplace"]["twilio_code"]
            # Optional: Hangup after listing? Set hangup_call = True
        elif any(kw in user_text_lower for kw in config.DEMO_RESPONSES["marketplace"]["triggers_status"]):
            # Check if the specific listing ID is mentioned (simple check)
            if '1009' in user_text_lower or '१०९' in user_text_lower: # Check for digits or Hindi word if needed
                 final_spoken_response = config.DEMO_RESPONSES["marketplace"]["response_status_success"]
                 response_lang_twilio = config.DEMO_RESPONSES["marketplace"]["twilio_code"]
                 hangup_call = True # Hang up after confirming payment
            else:
                 # If status asked but no ID, prompt for it (or use default response)
                 final_spoken_response = "कृपया अपनी लिस्टिंग आईडी बताएं जिसकी आप स्थिति जांचना चाहते हैं।" # Please tell the listing ID you want to check status for.
                 response_lang_twilio = config.DEMO_RESPONSES["marketplace"]["twilio_code"]

    # Prediction Demo (English) - Check if preferred lang is English
    elif lang_code == 'en':
         if any(kw in user_text_lower for kw in config.DEMO_RESPONSES["prediction"]["triggers_weather"]):
              final_spoken_response = config.DEMO_RESPONSES["prediction"]["response_weather"].format(location=loc.capitalize())
              response_lang_twilio = config.DEMO_RESPONSES["prediction"]["twilio_code"]
         elif any(kw in user_text_lower for kw in config.DEMO_RESPONSES["prediction"]["triggers_recommend"]):
              # Fetch soil type from context if available (Needs to be added if not already)
              soil_type = context.get('soil_type', 'the current') # Placeholder
              final_spoken_response = config.DEMO_RESPONSES["prediction"]["response_recommend"].format(location=loc.capitalize(), soil_type=soil_type)
              response_lang_twilio = config.DEMO_RESPONSES["prediction"]["twilio_code"]

    # Subsidy Demo (Marathi) - Check if preferred lang is Marathi
    elif lang_code == 'mr':
         if any(kw in user_text_lower for kw in config.DEMO_RESPONSES["subsidy"]["triggers_query"]):
              final_spoken_response = config.DEMO_RESPONSES["subsidy"]["response_list"]
              response_lang_twilio = config.DEMO_RESPONSES["subsidy"]["twilio_code"]
              # Don't hang up, wait for answer to follow-up question
         elif "नाही" in user_text_lower or "nako" in user_text_lower or "nahi" in user_text_lower: # User says no to more info
              final_spoken_response = config.DEMO_RESPONSES["subsidy"]["response_no_info"]
              response_lang_twilio = config.DEMO_RESPONSES["subsidy"]["twilio_code"]
              hangup_call = True # Hang up after saying ok

    # --- Fallback or Non-Demo Flow ---
    if final_spoken_response is None:
        logger.warning(f"No hardcoded demo match for user input: '{stt_res.text}'. Falling back to LLM (if available).")
        if llm_client_present:
             # If you want a generic LLM fallback, put the original LLM call logic here.
             # For the pure hardcoded demo, we can just say we didn't understand.
             final_spoken_response = "क्षमस्व, मला समजले नाही. आपण पुन्हा सांगू शकता का?" if lang_code == 'mr' else "Sorry, I didn't understand that. Could you please repeat?"
             response_lang_twilio = tw_code # Use preferred lang for fallback message
             # Decide if you want to hang up or gather again on fallback
             # hangup_call = True
        else:
             # No LLM and no demo match
             logger.error("LLM client not available and no demo match found. Cannot respond.")
             final_spoken_response = config.ERROR_MESSAGE
             response_lang_twilio = tw_code
             hangup_call = True # Hang up if completely stuck


    # --- Response Generation ---
    twiml: Optional[str] = None
    log_conversation_turn("USER", stt_res.text, lang=stt_res.language, call_sid=call_sid) # Log user turn
    cm.add_message_to_history(call_sid, 'user', stt_res.text) # Add user utterance to history

    if not final_spoken_response: # Should not happen if fallback is set
        error_message = config.ERROR_MESSAGE; logger.error(f"Error state for {call_sid[-4:]}: final_spoken_response is None"); log_conversation_turn("SYSTEM", f"Error: {error_message}", call_sid=call_sid, lang=lang_code)
        twiml = th.create_error_response(error_message, response_language=tw_code)
    elif hangup_call:
        log_conversation_turn("ASSISTANT", final_spoken_response, lang=lang_code, call_sid=call_sid); cm.add_message_to_history(call_sid, 'assistant', final_spoken_response); logger.info(f"Hanging up {call_sid[-4:]}")
        twiml = th.create_say_hangup_response(final_spoken_response, response_lang_twilio)
    else:
        log_conversation_turn("ASSISTANT", final_spoken_response, lang=lang_code, call_sid=call_sid); cm.add_message_to_history(call_sid, 'assistant', final_spoken_response)
        twiml = th.create_say_gather_response(final_spoken_response, '/process-speech', config.LISTEN_PROMPT, response_lang_twilio)

    if not twiml: logger.critical(f"TwiML None for {call_sid[-4:]}."); twiml = th.create_error_response(config.ERROR_MESSAGE, response_language=tw_code)
    return Response(twiml, mimetype='text/xml')


# ... ( /recording-status, /call-status routes) ...
@app.route("/recording-status", methods=['POST'])
def handle_recording_status(): call_sid=request.values.get('CallSid'); status=request.values.get('RecordingStatus'); url=request.values.get('RecordingUrl'); duration=request.values.get('RecordingDuration'); error_code=request.values.get('ErrorCode'); logger.info(f"RecStatus: Sid={call_sid[-4:]}, St={status}, Dur={duration}, URL={url}, Err={error_code}"); return Response(status=200)
@app.route("/call-status", methods=['POST'])
def handle_call_status():
    call_sid = request.values.get('CallSid')
    status = request.values.get('CallStatus')
    logger.info(f"CallStatus: Sid={call_sid}, St={status}")

    end_statuses = ['completed', 'failed', 'canceled', 'no-answer', 'busy']
    if status in end_statuses:
        logger.info(f"Call ended '{status}'. Clearing {call_sid}")
        clear_call_context(call_sid)
        cm.clear_conversation_history(call_sid)

    return Response(status=200)
    
# ... (__main__ block) ...
if __name__ == "__main__":
    # No backend check needed if hardcoding
    # if isinstance(bf, DummyBackend): logger.warning("Backend functions NOT loaded.")
    if not twilio_client: logger.warning("Twilio client NOT initialized.")
    if not stt_client: logger.error("STT Client FAILED to initialize.")
    if not llm_client_present: logger.warning("LLM Client NOT initialized. Only hardcoded flows active.")
    else: logger.info(f"LLM Client initialized: {config.LLM_PROVIDER}")
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

# --- END OF FILE app.py ---