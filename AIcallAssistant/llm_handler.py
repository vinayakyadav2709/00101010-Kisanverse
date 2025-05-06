# --- START OF FILE llm_handler.py ---

import logging
import sys
import config
from llm_selector import get_llm_client # Import the factory function

logger = logging.getLogger(__name__)

# Initialize the LLM client using the selector
llm_client = get_llm_client()

if not llm_client:
    logger.critical("LLM client initialization failed. Check config and connections.")
    # Depending on desired behavior, you might exit or continue with limited functionality
    # sys.exit("Exiting due to LLM client failure.")

def get_llm_response(messages, language_hint=None): # language_hint might be less critical now if language is in system prompt
    if llm_client:
        # The client wrapper should handle logging internally
        return llm_client.get_response(messages)
    else:
        logger.error("LLM client is not available.")
        return None

def get_system_prompt(location="an unspecified location", language="English (en)"):
    """
    Generates the initial system prompt using dynamic information.
    """
    prompt_content = config.get_dynamic_system_prompt(location, language)
    return {'role': 'system', 'content': prompt_content}

# --- END OF FILE llm_handler.py ---