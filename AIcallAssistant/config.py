# --- START OF FILE config.py ---

import os
from dotenv import load_dotenv
import logging
from typing import Dict

# Setup Logging FIRST
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

load_dotenv()

# --- LLM Configuration ---
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama').lower()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', "http://localhost:11434")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma3:4b-it-q8_0')
E2E_BASE_URL = os.getenv('E2E_BASE_URL')
E2E_API_KEY = os.getenv('E2E_API_KEY')
E2E_MODEL_NAME = os.getenv('E2E_MODEL_NAME')

# --- Shared LLM Settings ---
LLM_SYSTEM_PROMPT_TEMPLATE = """You are a concise, helpful agricultural assistant speaking to a farmer over a phone call.
The user is calling from {location}. Their preferred language is likely {language}.
- Keep answers brief and directly relevant to the user's query.
- If the user speaks a language other than English, respond ONLY in that language if possible, otherwise use English.
- You can use tools by outputting ONLY a JSON object: {{"tool_name": "function_name", "arguments": {{"arg1": "value1"}}}}.
- Available Tools:
    - add_crop_listing: Add a farmer's crop to sell. Args: farmer_id (str, use 'caller'), crop_type (str, required), price_per_kg (int, required), total_quantity (int, required).
    - get_marketplace_orders: Find buy/sell orders or check status. Args: listing_id (str, optional, use if checking status), product_name (str, optional).
    - get_weather_forecast: Get the weather for a location. Args: location (str, optional, defaults to user's location).
    - get_crop_recommendations: Get crop suggestions. Args: location (str, optional, defaults to user's location).
    - get_subsidy_info: Find government/NGO subsidies. Args: scheme_name (str, optional), location (str, optional, defaults to user's location).
    - get_contracts: Find available farming contracts. Args: crop_name (str, optional), company (str, optional), location (str, optional).
- When providing information obtained from a tool, synthesize it into a natural spoken answer IN THE USER'S LIKELY LANGUAGE ({language}). Do NOT output raw data or JSON in your final spoken response.
- If unsure or a tool fails, politely state you cannot fulfill the request in the user's language ({language}).
- Always remember the user's location is {location}. Use it as default for location arguments if not specified by the user.
- For the marketplace demo: First, use add_crop_listing. Later, if asked about status, use get_marketplace_orders with the simulated listing_id (you'll need to remember it or ask user). Acknowledge payment if the tool indicates fulfillment. Respond in Hindi.
- For the subsidy demo: Use get_subsidy_info. Present the results in Marathi. Ask the Marathi follow-up question: 'तुम्हाला यापैकी कोणत्याही योजनेबद्दल अधिक माहिती हवी आहे का?'
- For predictions demo: Use get_weather_forecast and get_crop_recommendations. Summarize results in English.



"""
CONVERSATION_HISTORY_LIMIT = 10

def get_dynamic_system_prompt(location="an unspecified location", language="English (en)"):
    return LLM_SYSTEM_PROMPT_TEMPLATE.format(location=location, language=language)

# --- Twilio / Voice Configuration ---

# --- MODIFIED: Language Specific Greetings ---
GREETING_TEMPLATES: Dict[str, str] = {
    'en': "Hello from your AI farming assistant! Calling from {location}, is that right? How can I help you today in English?",
    'hi': "नमस्ते! मैं आपका AI कृषि सहायक हूँ! {location} से कॉल कर रहे हैं, ठीक है? आज मैं आपकी हिंदी में कैसे मदद कर सकता हूँ?",
    'mr': "नमस्कार! मी तुमचा AI शेती सहाय्यक! {location} येथून कॉल करत आहात, बरोबर? आज मी तुमची मराठीत कशी मदत करू शकतो?",
    'kn': "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ AI ಕೃಷಿ ಸಹಾಯಕ! {location} ಇಂದ ಕರೆ ಮಾಡುತ್ತಿದ್ದೀರಾ, ಸರಿ ತಾನೇ? ಇಂದು ನಾನು ನಿಮಗೆ ಕನ್ನಡದಲ್ಲಿ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
    'ml': "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ AI കൃഷി സഹായിയാണ്! {location} എന്ന സ്ഥലത്ത് നിന്നാണോ വിളിക്കുന്നത്, ശരിയല്ലേ? ഇന്ന് നിങ്ങളെ മലയാളത്തിൽ എങ്ങനെ സഹായിക്കാൻ സാധിക്കും?",
    'ta': "வணக்கம்! நான் உங்கள் AI விவசாய உதவியாளர்! {location} இடத்திலிருந்து அழைக்கிறீர்கள், சரியா? இன்று நான் உங்களுக்கு தமிழில் எப்படி உதவ முடியும்?",
    'te': "నమస్కారం! నేను మీ AI వ్యవసాయ సహాయకుడిని! {location} నుండి కాల్ చేస్తున్నారు, సరియైనదేనా? ఈ రోజు నేను మీకు తెలుగులో ఎలా సహాయపడగలను?"
    # Add other supported languages here
}
# --- END MODIFIED ---

LISTEN_PROMPT = "Is there anything else I can help with?" # TODO: Translate this too if needed later
ERROR_MESSAGE = "Sorry, I encountered an error. Please try again later." # TODO: Translate
NO_SPEECH_MESSAGE = "Sorry, I didn't catch that. Could you please repeat?" # TODO: Translate
CONVERSATION_ERROR_MESSAGE = "Sorry, there was an issue tracking our conversation. Please call back." # TODO: Translate
DEFAULT_LANGUAGE_CODE = 'en-IN'
SPEECH_TIMEOUT = 'auto'
PROFANITY_FILTER = False

# --- STT Configuration ---
STT_PROVIDER = os.getenv('STT_PROVIDER', 'whisper').lower()
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'turbo')
E2E_TIR_ACCESS_TOKEN = os.getenv('E2E_TIR_ACCESS_TOKEN')
E2E_TIR_API_KEY = os.getenv('E2E_TIR_API_KEY')
E2E_TIR_PROJECT_ID = os.getenv('E2E_TIR_PROJECT_ID')
E2E_TIR_TEAM_ID = os.getenv('E2E_TIR_TEAM_ID')
E2E_WHISPER_MODEL_NAME = os.getenv('E2E_WHISPER_MODEL_NAME', 'whisper-large-v3')

# --- Twilio Credentials ---
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# --- Location to Language Mapping ---
LOCATION_TO_LANGUAGE: Dict[str, Dict[str, str]] = {
    'mumbai':    {'lang_code': 'mr', 'lang_name': 'Marathi', 'twilio_code': 'mr-IN'},
    'pune':      {'lang_code': 'mr', 'lang_name': 'Marathi', 'twilio_code': 'mr-IN'},
    'maharashtra':{'lang_code': 'mr', 'lang_name': 'Marathi', 'twilio_code': 'mr-IN'},
    'delhi':     {'lang_code': 'hi', 'lang_name': 'Hindi', 'twilio_code': 'hi-IN'},
    'bangalore': {'lang_code': 'kn', 'lang_name': 'Kannada', 'twilio_code': 'kn-IN'},
    'mysore':    {'lang_code': 'kn', 'lang_name': 'Kannada', 'twilio_code': 'kn-IN'},
    'karnataka': {'lang_code': 'kn', 'lang_name': 'Kannada', 'twilio_code': 'kn-IN'},
    'kochi':     {'lang_code': 'ml', 'lang_name': 'Malayalam', 'twilio_code': 'ml-IN'},
    'kerala':    {'lang_code': 'ml', 'lang_name': 'Malayalam', 'twilio_code': 'ml-IN'},
    'chennai':   {'lang_code': 'ta', 'lang_name': 'Tamil', 'twilio_code': 'ta-IN'},
    'tamil nadu':{'lang_code': 'ta', 'lang_name': 'Tamil', 'twilio_code': 'ta-IN'},
    'hyderabad': {'lang_code': 'te', 'lang_name': 'Telugu', 'twilio_code': 'te-IN'},
    'andhra pradesh': {'lang_code': 'te', 'lang_name': 'Telugu', 'twilio_code': 'te-IN'},
    'telangana': {'lang_code': 'te', 'lang_name': 'Telugu', 'twilio_code': 'te-IN'},
    'chhattisgarh':{'lang_code': 'hi', 'lang_name': 'Hindi', 'twilio_code': 'hi-IN'},
    'madhya pradesh':{'lang_code': 'hi', 'lang_name': 'Hindi', 'twilio_code': 'hi-IN'},
    'odisha':    {'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
    'uttar pradesh':{'lang_code': 'hi', 'lang_name': 'Hindi', 'twilio_code': 'hi-IN'},
    'jharkhand': {'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
    'bihar':     {'lang_code': 'hi', 'lang_name': 'Hindi', 'twilio_code': 'hi-IN'},
    'gujarat':   {'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
    'kolkata':   {'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
    'west bengal':{'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
    'default':   {'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
    'unknown':   {'lang_code': 'en', 'lang_name': 'English', 'twilio_code': 'en-IN'},
}

# --- Simulator Configuration ---
SIMULATOR_APP_URL = "http://127.0.0.1:5000"

# --- Log Configuration Info ---
# ... (Logging remains the same) ...
logger.info(f"LLM_PROVIDER: {LLM_PROVIDER}")
if LLM_PROVIDER == 'ollama': logger.info(f"OLLAMA_HOST/MODEL: {OLLAMA_HOST} / {OLLAMA_MODEL}")
elif LLM_PROVIDER == 'e2e':
    logger.info(f"E2E_BASE_URL: {E2E_BASE_URL}"); logger.info(f"E2E_MODEL_NAME: {E2E_MODEL_NAME}")
    if not E2E_API_KEY: logger.warning("E2E_API_KEY missing!");
    if not E2E_BASE_URL: logger.warning("E2E_BASE_URL missing!");
    if not E2E_MODEL_NAME: logger.warning("E2E_MODEL_NAME missing!")
logger.info(f"STT_PROVIDER: {STT_PROVIDER}")
if STT_PROVIDER == 'whisper': logger.info(f"WHISPER_MODEL_SIZE: {WHISPER_MODEL_SIZE}")
elif STT_PROVIDER == 'e2e_whisper':
    logger.info(f"E2E_WHISPER_MODEL_NAME: {E2E_WHISPER_MODEL_NAME}")
    if not all([E2E_TIR_ACCESS_TOKEN, E2E_TIR_API_KEY, E2E_TIR_PROJECT_ID, E2E_TIR_TEAM_ID]): logger.warning("E2E Whisper STT credentials missing!")
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN: logger.warning("TWILIO credentials missing!")
else: logger.info("Twilio credentials found.")

# --- END OF FILE config.py ---