# --- START OF FILE twilio_handler.py ---

import logging
from twilio.twiml.voice_response import VoiceResponse, Gather, Hangup # <-- Import Hangup
from config import (
    DEFAULT_LANGUAGE_CODE, SPEECH_TIMEOUT, PROFANITY_FILTER,
    STT_PROVIDER, LISTEN_PROMPT, LOCATION_TO_LANGUAGE
)
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ... (LANGUAGE_MAP_TWILIO and map_language_code_to_twilio remain the same) ...
LANGUAGE_MAP_TWILIO: Dict[str, str] = { info['lang_code']: info['twilio_code'] for _, info in LOCATION_TO_LANGUAGE.items() if 'lang_code' in info and 'twilio_code' in info }
if 'en' not in LANGUAGE_MAP_TWILIO: LANGUAGE_MAP_TWILIO['en'] = 'en-IN'
logger.debug(f"Generated Twilio Language Map: {LANGUAGE_MAP_TWILIO}")
def map_language_code_to_twilio(lang_code_short: Optional[str], default_fallback: str = DEFAULT_LANGUAGE_CODE) -> str:
    if not lang_code_short: return default_fallback
    mapped_code = LANGUAGE_MAP_TWILIO.get(lang_code_short, default_fallback)
    if mapped_code == default_fallback and lang_code_short not in LANGUAGE_MAP_TWILIO: logger.warning(f"Could not map lang '{lang_code_short}'. Used fallback: {default_fallback}")
    return mapped_code

# ... (_create_gather, create_greeting_gather_response, create_say_gather_response, create_no_speech_response, create_error_response remain the same) ...
def _create_gather(action_url: str, method: str = 'POST', gather_language: Optional[str] = None) -> Gather:
    effective_gather_lang = gather_language or DEFAULT_LANGUAGE_CODE; gather_options = {'input': 'speech', 'action': action_url, 'method': method, 'speechTimeout': SPEECH_TIMEOUT, 'language': effective_gather_lang, 'profanityFilter': PROFANITY_FILTER, 'actionOnEmptyResult': True }
    if STT_PROVIDER == 'whisper': gather_options['record'] = 'record-from-answer'; gather_options['recordingStatusCallback'] = '/recording-status'; gather_options['recordingStatusCallbackMethod'] = 'POST'; logger.info(f"Enabling recording ({gather_options['record']}) for Gather. Lang: {effective_gather_lang}")
    return Gather(**gather_options)
def create_greeting_gather_response(greeting_message: str, gather_action_url: str, greeting_language: Optional[str] = None) -> str:
    response = VoiceResponse(); say_lang_twilio = greeting_language or DEFAULT_LANGUAGE_CODE; logger.info(f"Generating greeting lang: {say_lang_twilio}"); response.say(greeting_message, language=say_lang_twilio); gather = _create_gather(gather_action_url, gather_language=say_lang_twilio); response.append(gather); logger.debug(f"Generated Greeting/Gather TwiML"); return str(response)
def create_say_gather_response(message_to_say: str, gather_action_url: str, listen_prompt: str, response_language: Optional[str] = None) -> str:
    response = VoiceResponse(); say_lang_twilio = response_language or DEFAULT_LANGUAGE_CODE
    if not message_to_say or not message_to_say.strip(): logger.warning("Empty message to Say+Gather.")
    else: logger.info(f"Generating assistant response lang: {say_lang_twilio}"); response.say(message_to_say, language=say_lang_twilio)
    gather = _create_gather(gather_action_url, gather_language=say_lang_twilio)
    if listen_prompt and listen_prompt.strip(): logger.debug(f"Adding listen prompt."); gather.say(listen_prompt, language=say_lang_twilio)
    else: logger.warning("Listen prompt empty.")
    response.append(gather); logger.debug(f"Generated Say/Gather TwiML"); return str(response)
def create_no_speech_response(no_speech_message: str, gather_action_url: str, listen_prompt: str, response_language: Optional[str] = None) -> str:
    response = VoiceResponse(); say_lang_twilio = response_language or DEFAULT_LANGUAGE_CODE; logger.info(f"Generating 'no speech' response lang: {say_lang_twilio}"); response.say(no_speech_message, language=say_lang_twilio)
    gather = _create_gather(gather_action_url, gather_language=say_lang_twilio)
    if listen_prompt and listen_prompt.strip(): logger.debug(f"Adding listen prompt."); gather.say(listen_prompt, language=say_lang_twilio)
    response.append(gather); logger.debug(f"Generated No-Speech TwiML"); return str(response)
def create_error_response(error_message: str, hangup: bool = True, response_language: Optional[str] = None) -> str:
    response = VoiceResponse(); say_lang_twilio = response_language or DEFAULT_LANGUAGE_CODE; safe_error_message = error_message if error_message and error_message.strip() else "An unspecified error occurred."; logger.info(f"Generating error response lang: {say_lang_twilio}"); response.say(safe_error_message, language=say_lang_twilio)
    if hangup: logger.info("Adding Hangup to error response."); response.hangup()
    logger.debug(f"Generated Error TwiML (Hangup: {hangup})"); return str(response)


# --- ADDED MISSING FUNCTION ---
def create_say_hangup_response(message_to_say: str, response_language: Optional[str] = None) -> str:
    """Creates TwiML to say a final message and then hang up."""
    response = VoiceResponse()
    say_lang_twilio = response_language or DEFAULT_LANGUAGE_CODE

    if not message_to_say or not message_to_say.strip():
        logger.warning("Attempted to Say empty message before hangup. Hanging up directly.")
    else:
        logger.info(f"Generating final assistant message before hangup in language: {say_lang_twilio}")
        response.say(message_to_say, language=say_lang_twilio)

    logger.info("Adding Hangup verb to TwiML.")
    response.hangup() # Use the imported Hangup verb
    logger.debug(f"Generated Say/Hangup TwiML (Say Lang: {say_lang_twilio})")
    return str(response)
# --- END ADDED MISSING FUNCTION ---

def extract_call_data(request_values: Dict) -> Tuple[Optional[str], str, Optional[str]]:
    """Extracts key data points from the incoming Twilio request values."""
    call_sid = request_values.get('CallSid'); speech_result = request_values.get('SpeechResult', '').strip()
    recording_url = request_values.get('RecordingUrl'); recording_duration = request_values.get('RecordingDuration')
    logger.debug(f"Extracted CallSid={call_sid}, RecordingUrl={recording_url}, Duration={recording_duration}, (SpeechResult='{speech_result}')")
    return call_sid, speech_result, recording_url

# --- END OF FILE twilio_handler.py ---