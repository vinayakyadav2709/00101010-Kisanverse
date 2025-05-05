# --- START OF FILE stt_selector.py ---
import logging
import config
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class STTClientInterface:
    """Base interface for STT clients."""
    def transcribe_audio(self, audio_path_or_url: str, expected_language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Transcribes audio from a given path or URL.

        Args:
            audio_path_or_url (str): Local file path (no scheme) or URL.
            expected_language (Optional[str]): Language hint (short code like 'en', 'hi').
                                             Behavior depends on implementation (force vs hint).

        Returns:
            Optional[Dict[str, Any]]: A dictionary with 'text', 'language', 'is_reliable',
                                      or None on critical failure.
        """
        raise NotImplementedError

# --- Wrapper Selector ---
_stt_client_instance: Optional[STTClientInterface] = None

def get_stt_client() -> Optional[STTClientInterface]:
    """Factory function to get the configured STT client instance."""
    global _stt_client_instance
    if _stt_client_instance is None:
        provider = config.STT_PROVIDER.lower()
        logger.info(f"Initializing STT Client for provider: {provider}")
        if provider == 'whisper':
            from stt_handler import LocalWhisperWrapper # Import locally to avoid circular deps
            _stt_client_instance = LocalWhisperWrapper()
        elif provider == 'e2e_whisper':
            from stt_handler import E2EWhisperWrapper # Import locally
            _stt_client_instance = E2EWhisperWrapper()
        else:
            logger.error(f"Unsupported STT_PROVIDER configured: {config.STT_PROVIDER}")
            return None

        # Check if initialization failed internally within the wrapper
        if hasattr(_stt_client_instance, '_initialized') and not _stt_client_instance._initialized:
             logger.error(f"STT Client Initialization failed for provider '{provider}'. Check logs and configuration.")
             _stt_client_instance = None # Reset if failed

    return _stt_client_instance

# --- END OF FILE stt_selector.py ---