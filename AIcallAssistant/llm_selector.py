# --- START OF FILE llm_selector.py ---

import logging
import ollama
from openai import OpenAI  # For E2E compatibility using OpenAI SDK pattern
import config
from typing import Optional, List, Dict, Any # Added Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# --- Base Interface (Optional but good practice) ---
class LLMClientInterface:
    def get_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Takes a list of message dicts, returns the LLM response string or None."""
        raise NotImplementedError

# --- Ollama Client Wrapper ---
class OllamaClientWrapper(LLMClientInterface):
    def __init__(self, model=config.OLLAMA_MODEL, host=config.OLLAMA_HOST):
        self.model = model
        self.host = host
        self.client = None
        self._connect()

    def _connect(self):
        try:
            self.client = ollama.Client(host=self.host)
            self.client.list() # Test connection
            logger.info(f"Successfully connected to Ollama ({self.host}). Using model: {self.model}")
        except Exception as e:
            logger.error(f"CRITICAL: Could not connect to Ollama ({self.host}): {e}")
            # Decide if you want to raise an error or let it fail later
            # raise ConnectionError(f"Failed to connect to Ollama: {e}")
            self.client = None # Ensure client is None if connection failed

    def get_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        if not self.client:
            logger.error("Ollama client not initialized or connection failed.")
            return None
        try:
            # Ensure messages are in the format Ollama expects
            # Make sure content is stringified, as Pydantic models might be passed
            formatted_messages = [{'role': msg['role'], 'content': str(msg['content'])} for msg in messages]

            logger.info(f"Sending to Ollama (History length: {len(formatted_messages)} messages)...")
            # logger.debug(f"Ollama Request Messages: {formatted_messages}") # Uncomment for deep debug
            response = self.client.chat(
                model=self.model,
                messages=formatted_messages
                # Add options like temperature if needed: options={'temperature': 0.7}
            )
            assistant_response = response['message']['content'].strip()
            # logger.debug(f"Ollama Raw Response Obj: {response}") # Uncomment for deep debug
            logger.info(f"Ollama Raw Response Text: '{assistant_response}'")
            return assistant_response
        except Exception as e:
            logger.error(f"Error interacting with Ollama: {e}", exc_info=True)
            return None

# --- E2E Networks Client Wrapper (Using OpenAI SDK) ---
class E2EClientWrapper(LLMClientInterface):
    def __init__(self, base_url=config.E2E_BASE_URL, api_key=config.E2E_API_KEY, model=config.E2E_MODEL_NAME):
        if not base_url or not api_key or not model:
            logger.error("E2E Networks configuration (URL, API Key, Model Name) is missing in config/env.")
            self.client = None
            self.model = None
            return

        self.model = model
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            # We can't easily "test connection" like with ollama.list() without making a real call
            # A lightweight call like listing models might work if the API supports it, otherwise assume init is ok.
            # self.client.models.list() # Example: This might work or fail depending on E2E's OpenAI compatibility
            logger.info(f"E2E Networks Client Initialized. Base URL: {base_url}, Model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize E2E Networks client: {e}")
            self.client = None

    def get_response(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        if not self.client or not self.model:
            logger.error("E2E client not initialized or configuration missing.")
            return None
        try:
            # Ensure messages are in the format OpenAI SDK expects
            # Filter out 'tool' role if E2E model doesn't support it directly
            # Or adapt the format if needed based on E2E documentation
            # Make sure content is stringified
            formatted_messages = [{'role': msg['role'], 'content': str(msg['content'])}
                                  for msg in messages if msg['role'] in ['system', 'user', 'assistant']]
            # Note: If E2E *does* support tool calling via the OpenAI API structure,
            # you might need to include the 'tool' role messages differently or format tools argument.

            logger.info(f"Sending to E2E Networks (History length: {len(formatted_messages)} messages)...")
            # logger.debug(f"E2E Request Messages: {formatted_messages}") # Uncomment for deep debug
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=0.5, # Adjust parameters as needed based on E2E model behavior
                max_tokens=1024, # Adjust max response length as needed
                # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0,
                stream=False # Get the full response at once for simplicity here
            )
            # logger.debug(f"E2E Raw Response Obj: {completion}") # Uncomment for deep debug
            assistant_response = completion.choices[0].message.content.strip()
            logger.info(f"E2E Raw Response Text: '{assistant_response}'")
            return assistant_response
        except Exception as e:
            logger.error(f"Error interacting with E2E Networks API: {e}", exc_info=True)
            return None


# --- Factory Function ---
def get_llm_client() -> Optional[LLMClientInterface]:
    """Creates and returns the appropriate LLM client based on config."""
    provider = config.LLM_PROVIDER.lower()
    logger.info(f"Attempting to initialize LLM Client for provider: {provider}")

    client: Optional[LLMClientInterface] = None
    if provider == 'ollama':
        client = OllamaClientWrapper()
    elif provider == 'e2e':
        client = E2EClientWrapper()
    # Add more providers here if needed
    # elif provider == 'some_other_service':
    #     client = SomeOtherServiceClientWrapper()
    else:
        logger.error(f"Unsupported LLM_PROVIDER configured: {config.LLM_PROVIDER}")
        return None # Explicitly return None for unsupported provider

    # Check if the client failed to initialize (e.g., missing E2E config or Ollama connection fail)
    # Check the internal 'client' attribute which should be None if init failed.
    initialization_failed = False
    if isinstance(client, E2EClientWrapper) and client.client is None:
        initialization_failed = True
    elif isinstance(client, OllamaClientWrapper) and client.client is None:
        initialization_failed = True
    # Add checks for other client types here if needed

    if initialization_failed:
         logger.error(f"Initialization failed for LLM provider '{provider}'. Check logs and configuration.")
         return None # Return None if the selected provider's client object is None

    logger.info(f"LLM Client for provider '{provider}' initialized successfully.")
    return client

# --- END OF FILE llm_selector.py ---