# --- START OF FILE stt_handler.py ---

import logging
import os
import requests
import tempfile
import urllib.parse
from config import (
    WHISPER_MODEL_SIZE, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, DEFAULT_LANGUAGE_CODE,
    E2E_TIR_ACCESS_TOKEN, E2E_TIR_API_KEY, E2E_TIR_PROJECT_ID, E2E_TIR_TEAM_ID,
    E2E_WHISPER_MODEL_NAME
)
from typing import Optional, Dict, Any
from stt_selector import STTClientInterface # Import base class

logger = logging.getLogger(__name__)

# --- Helper for Downloading ---
def _download_audio(recording_url: str) -> Optional[str]:
    """Downloads audio from Twilio URL or verifies local path."""
    file_to_delete = None
    transcription_input_path = None
    try:
        if recording_url.startswith('file://'):
            parsed_url = urllib.parse.urlparse(recording_url)
            local_audio_path_raw = urllib.parse.unquote(parsed_url.path)
            if os.name == 'nt' and local_audio_path_raw.startswith('/'): local_audio_path = local_audio_path_raw[1:]
            else: local_audio_path = local_audio_path_raw
            if not os.path.exists(local_audio_path): logger.error(f"Local audio file not found: {local_audio_path}"); return None
            if os.path.getsize(local_audio_path) < 500: logger.warning(f"Local audio {local_audio_path} small."); return None # Indicate empty but not critical failure? Or return path? Let's return path for now.
            transcription_input_path = local_audio_path
            logger.info(f"Using local audio file: {transcription_input_path}")
        else:
            if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN: logger.error("Twilio credentials missing for download."); return None
            logger.info(f"Attempting download: {recording_url}")
            if not any(recording_url.lower().endswith(ext) for ext in ['.wav', '.mp3']): url_mp3, url_wav = recording_url + ".mp3", recording_url + ".wav"; url_to_try = url_mp3
            else: url_to_try = recording_url; url_wav = recording_url.replace('.mp3', '.wav') # Ensure wav fallback possible
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN); response = None
            for url_attempt in [url_to_try, url_wav]: # Try MP3 then WAV
                 if response and response.ok: break
                 logger.debug(f"Trying download from: {url_attempt}")
                 try:
                      response = requests.get(url_attempt, auth=auth, stream=True, timeout=30)
                      if response.ok: logger.info(f"Connected OK to {url_attempt}"); break
                      else: logger.warning(f"Download attempt failed {url_attempt} (Status: {response.status_code})")
                 except requests.exceptions.RequestException as e_req: logger.warning(f"Request failed {url_attempt}: {e_req}"); continue
            if not response or not response.ok: logger.error(f"Download failed for: {recording_url}"); return None
            content_type = response.headers.get('Content-Type', '').lower()
            if 'audio/mpeg' in content_type or url_attempt.endswith('.mp3'): suffix = '.mp3'
            elif 'audio/wav' in content_type or 'audio/x-wav' in content_type or url_attempt.endswith('.wav'): suffix = '.wav'
            else: suffix = '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                size = 0; [temp_audio.write(chunk) for chunk in response.iter_content(chunk_size=8192) if (size:=size+len(chunk))] # Write chunks & track size
                temp_downloaded_path = temp_audio.name
                file_to_delete = temp_downloaded_path
            if size < 500: logger.warning(f"Downloaded audio {temp_downloaded_path} small."); # Return path even if small
            transcription_input_path = temp_downloaded_path
            logger.info(f"Downloaded ({size} bytes, suffix: {suffix}) to {transcription_input_path}")

        return transcription_input_path

    except Exception as e:
        logger.error(f"Error during audio download/handling for {recording_url}: {e}", exc_info=True)
        return None
    finally:
        # Caller's responsibility to delete the temp file if one was created
        # This function returns the path (temp or original) or None
        pass

# --- Local Whisper Implementation ---
class LocalWhisperWrapper(STTClientInterface):
    def __init__(self):
        self.model = None
        self._initialized = False
        try:
            import whisper
            logger.info(f"Attempting to load local Whisper model: {WHISPER_MODEL_SIZE}")
            # Explicitly use CPU unless GPU is intended/tested
            self.model = whisper.load_model(WHISPER_MODEL_SIZE, device='cpu')
            logger.info(f"Local Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully using CPU.")
            self._initialized = True
        except ImportError:
            logger.error("`openai` library not installed or whisper component missing. Local Whisper STT unavailable.")
        except Exception as e:
            logger.error(f"Failed to load local Whisper model '{WHISPER_MODEL_SIZE}': {e}", exc_info=True)

    def transcribe_audio(self, audio_path_or_url: str, expected_language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self._initialized or not self.model:
            logger.error("Local Whisper model not loaded. Cannot transcribe.")
            return None

        default_lang = expected_language or DEFAULT_LANGUAGE_CODE.split('-')[0]
        result = {"text": "", "language": default_lang, "confidence": None, "is_reliable": False}
        audio_path = None
        temp_file_to_delete = None

        try:
            # Download or verify path
            logger.debug("LocalWhisper: Getting audio path...")
            input_path = _download_audio(audio_path_or_url)
            if not input_path:
                 logger.error("LocalWhisper: Failed to get valid audio path.")
                 return None # Critical failure if path invalid

            # Check if download created a temp file
            if input_path.startswith(tempfile.gettempdir()):
                 temp_file_to_delete = input_path
            audio_path = input_path

            # Handle potentially small/empty files returned by _download_audio
            if os.path.exists(audio_path) and os.path.getsize(audio_path) < 500:
                 logger.warning(f"LocalWhisper: Audio file {audio_path} too small. Returning unreliable.")
                 result['is_reliable'] = False
                 return result # Not a critical failure, just empty

            logger.info(f"LocalWhisper: Transcribing '{os.path.basename(audio_path)}', Expected Lang: {expected_language or 'Auto'}")
            whisper_result = self.model.transcribe(
                audio_path,
                fp16=False,
                language=expected_language # Force language if provided
            )

            result['text'] = whisper_result.get('text', '').strip()
            detected_language = whisper_result.get('language')
             # Use expected if forced, otherwise detected, otherwise default
            result['language'] = expected_language if expected_language else (detected_language if detected_language else default_lang)
            result['is_reliable'] = bool(result['text'])
            result['confidence'] = None

            logger.info(f"LocalWhisper: Transcription complete. Reported Lang: {result['language']}, Reliable: {result['is_reliable']}")
            log_text = result['text'][:100] + ('...' if len(result['text']) > 100 else '')
            logger.info(f"LocalWhisper: Text (preview): '{log_text}'")
            return result

        except Exception as e:
            logger.error(f"LocalWhisper: Error during transcription: {e}", exc_info=True)
            return None
        finally:
            # Clean up temp file if one was created by _download_audio
            if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                try: os.remove(temp_file_to_delete); logger.debug(f"LocalWhisper: Deleted temp file {temp_file_to_delete}")
                except OSError as e_del: logger.error(f"LocalWhisper: Error deleting temp file {temp_file_to_delete}: {e_del}")


# --- E2E Whisper Implementation ---
class E2EWhisperWrapper(STTClientInterface):
    def __init__(self):
        self.client = None
        self.model_name = config.E2E_WHISPER_MODEL_NAME
        self._initialized = False
        if not all([config.E2E_TIR_ACCESS_TOKEN, config.E2E_TIR_API_KEY, config.E2E_TIR_PROJECT_ID, config.E2E_TIR_TEAM_ID]):
            logger.error("E2E Whisper selected, but required E2E_TIR credentials missing in config/env.")
            return
        try:
            # Set environment variables for the library
            os.environ['E2E_TIR_ACCESS_TOKEN'] = config.E2E_TIR_ACCESS_TOKEN
            os.environ['E2E_TIR_API_KEY'] = config.E2E_TIR_API_KEY
            os.environ['E2E_TIR_PROJECT_ID'] = config.E2E_TIR_PROJECT_ID
            os.environ['E2E_TIR_TEAM_ID'] = config.E2E_TIR_TEAM_ID

            from e2enetworks.cloud import tir # Import after setting env vars
            tir.init() # Initialize E2E library
            self.client = tir.ModelAPIClient()
            logger.info(f"E2E Whisper client initialized. Using model: {self.model_name}")
            self._initialized = True
        except ImportError:
             logger.error("`e2enetworks` library not installed. Please install it (`pip install e2enetworks`). E2E Whisper unavailable.")
        except Exception as e:
            logger.error(f"Failed to initialize E2E Whisper client: {e}", exc_info=True)

    def transcribe_audio(self, audio_path_or_url: str, expected_language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self._initialized or not self.client:
            logger.error("E2E Whisper client not initialized. Cannot transcribe.")
            return None

        default_lang = expected_language or DEFAULT_LANGUAGE_CODE.split('-')[0]
        result = {"text": "", "language": default_lang, "confidence": None, "is_reliable": False}
        audio_path = None
        temp_file_to_delete = None

        try:
            # Download or verify path
            logger.debug("E2EWhisper: Getting audio path...")
            input_path = _download_audio(audio_path_or_url)
            if not input_path:
                 logger.error("E2EWhisper: Failed to get valid audio path.")
                 return None

            if input_path.startswith(tempfile.gettempdir()):
                 temp_file_to_delete = input_path
            audio_path = input_path

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 500:
                 logger.warning(f"E2EWhisper: Audio file {audio_path} missing or too small. Returning unreliable.")
                 result['is_reliable'] = False
                 # Clean up if temp file was created for this small download
                 if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                      try: os.remove(temp_file_to_delete)
                      except OSError: pass
                 return result

            logger.info(f"E2EWhisper: Transcribing '{os.path.basename(audio_path)}' using model '{self.model_name}', Lang Hint: {expected_language or 'Auto'}")

            # Prepare data for E2E API
            # The E2E library expects the *file path* for inference
            data = {
                "input": audio_path, # Pass the path to the local audio file
                "language": expected_language if expected_language else "auto", # Use 'auto' or provide hint
                "task": "transcribe",
                # "max_new_tokens": 400, # Optional: adjust if needed
                "return_timestamps": "none" # Or 'word'/'segment' if needed
            }

            logger.debug(f"E2E Whisper API request data: {data}")
            output = self.client.infer(model_name=self.model_name, data=data)
            logger.debug(f"E2E Whisper API raw output: {output}")

            # --- Parse E2E Response ---
            # Adjust parsing based on the actual structure of the 'output' object/dict
            # Assuming output might look like: {'text': '...', 'language': '...'} or similar
            if isinstance(output, dict):
                transcribed_text = output.get('text', '').strip()
                detected_language = output.get('language', default_lang) # Get detected lang if available

                result['text'] = transcribed_text
                # Use detected language if provided and valid, otherwise stick to default/expected
                result['language'] = detected_language if detected_language else default_lang
                result['is_reliable'] = bool(result['text'])
                # E2E might provide confidence - check output structure
                result['confidence'] = output.get('confidence') # Example

                logger.info(f"E2EWhisper: Transcription complete. Reported Lang: {result['language']}, Reliable: {result['is_reliable']}")
                log_text = result['text'][:100] + ('...' if len(result['text']) > 100 else '')
                logger.info(f"E2EWhisper: Text (preview): '{log_text}'")
            else:
                 logger.error(f"E2EWhisper: Unexpected output format from API: {type(output)}")
                 return None # Indicate failure if output is not parseable

            return result

        except Exception as e:
            logger.error(f"E2EWhisper: Error during transcription: {e}", exc_info=True)
            return None
        finally:
            # Clean up temp file if one was created
            if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                try: os.remove(temp_file_to_delete); logger.debug(f"E2EWhisper: Deleted temp file {temp_file_to_delete}")
                except OSError as e_del: logger.error(f"E2EWhisper: Error deleting temp file {temp_file_to_delete}: {e_del}")


# --- END OF FILE stt_handler.py ---