# --- START OF FILE twilio_simulator.py ---

import requests
import os
# import argparse # No longer needed
import time
import uuid
from xml.etree import ElementTree as ET
import logging
import tempfile
import queue # Ensure queue is imported
import urllib.parse
from typing import Optional, List, Dict, Any
import wave
import threading
import pathlib

# --- Audio Libraries / TTS ---
try:
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write as write_wav, read as read_wav
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    logging.warning("sounddevice, numpy, or scipy not found. Microphone input disabled. Audio merging might fail.")
    SOUNDDEVICE_AVAILABLE = False
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    logging.warning("gTTS not installed. Assistant audio output disabled.")
    TTS_AVAILABLE = False

# --- Config / Handler Imports ---
from config import ( SIMULATOR_APP_URL, LOGGING_LEVEL, LOGGING_FORMAT,
                    DEFAULT_LANGUAGE_CODE, LOCATION_TO_LANGUAGE )
from twilio_handler import LANGUAGE_MAP_TWILIO, map_language_code_to_twilio

# --- Logging Setup ---
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
logging.getLogger("requests").setLevel(logging.WARNING); logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
if TTS_AVAILABLE: logging.getLogger("gtts").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_MERGED_FILENAME_TEMPLATE = "merged_conversation_{call_sid}.wav"

# --- Helper Functions ---
# ... (log_conversation_turn, speak_text, find_verb_text, find_verb_attribute) ...
def log_conversation_turn(role: str, text: str, lang: Optional[str] = None, call_sid: Optional[str] = None, loc: Optional[str] = None): sid_info = f"[CallSid: {call_sid}]" if call_sid else "[SIMULATOR]"; lang_info = f"(Lang: {lang})" if lang else ""; loc_info = f"(Loc: {loc})" if loc else ""; print(f"\n{'='*10} {role.upper()} {sid_info}{loc_info} {'='*10}"); print(f"{text} {lang_info}"); print(f"{'='* (22 + len(role) + len(sid_info) + len(loc_info))}\n"); log_text = text[:500] + ('...' if len(text) > 500 else ''); logger.info(f"CONV_TURN {sid_info}{loc_info}: Role={role}, Lang={lang}, Text='{log_text}'")
def speak_text(text: str, lang: str = 'en', temp_dir: Optional[str] = None) -> Optional[str]:
    if not text or not text.strip() or not TTS_AVAILABLE: return None
    safe_text_prefix = "".join(c for c in text[:20] if c.isalnum() or c in " _-").strip() or "tts"; temp_tts_path = None
    try:
        if len(lang) > 2 and '-' in lang: lang = lang.split('-')[0]
        logger.info(f"Generating TTS for: '{text[:50]}...' (lang={lang})"); tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix=f"{safe_text_prefix}_", dir=temp_dir) as fp: temp_tts_path = fp.name
        tts.save(temp_tts_path); logger.info(f"Saved TTS audio to {temp_tts_path}")
        logger.info(f"Playing TTS audio via mpg123..."); cmd = f"mpg123 -q \"{temp_tts_path}\""; logger.debug(f"Executing command: {cmd}"); status = os.system(cmd); logger.info(f"mpg123 playback finished (status: {status}).")
        if status != 0: logger.warning(f"mpg123 command potentially failed (exit status: {status}).")
        return temp_tts_path
    except ImportError: logger.warning("gTTS not installed."); return None
    except ValueError as ve: logger.error(f"gTTS error, likely invalid language '{lang}': {ve}");
    except Exception as e: logger.error(f"Error during TTS: {e}", exc_info=True)
    if temp_tts_path and os.path.exists(temp_tts_path): 
        try: os.remove(temp_tts_path) 
        except OSError: pass
    return None
def find_verb_text(twiml_root: ET.Element, verb_tag: str) -> Optional[str]: verb = twiml_root.find(verb_tag); return verb.text.strip() if verb is not None and verb.text else None
def find_verb_attribute(twiml_root: ET.Element, verb_tag: str, attribute_name: str) -> Optional[str]: verb = twiml_root.find(verb_tag); return verb.get(attribute_name) if verb is not None else None

# --- Audio Recording ---
def record_audio(filename: str, stop_event: threading.Event) -> bool:
    """Records audio from microphone until stop_event is set."""
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("Sounddevice not available, cannot record audio.")
        return False
    q: queue.Queue[np.ndarray] = queue.Queue() # Initialize queue outside try

    def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        if status:
            logger.warning(f"Sounddevice recording status: {status}")
        q.put(indata.copy()) # Add data to the queue

    stream = None # Initialize stream variable
    recording_data: List[np.ndarray] = [] # Initialize recording_data list

    try:
        default_input_device = sd.query_devices(kind='input')
        if not default_input_device:
            logger.error("No audio input device found by sounddevice.")
            return False
        samplerate = SAMPLE_RATE
        logger.info(f"Attempting to record using input device: {default_input_device['name']} at fixed rate {samplerate} Hz")

        # Ensure the file exists for writing binary
        with open(filename, 'wb') as f:
             f.write(b'')

        print("\n[SIMULATOR] Press Enter to start recording...")
        input()
        print("[SIMULATOR] Recording... Press Enter again to stop.")
        stop_event.clear()

        # Start the input stream using a context manager
        with sd.InputStream(
            samplerate=samplerate,
            channels=CHANNELS,
            callback=audio_callback,
            dtype='int16'
        ) as stream: # Assign to stream variable within context
            # Wait for Enter key press to stop recording
            input()
            stop_event.set() # Signal recording to stop (though Enter already blocked)

        print("[SIMULATOR] Recording stopped.")

        # --- FIX: Retrieve data AFTER the stream context manager exits ---
        # This ensures the callback has finished processing blocks
        while not q.empty():
            recording_data.append(q.get())
        # --- END FIX ---

        if not recording_data:
            logger.warning("No audio data captured in the queue.")
            try: os.remove(filename)
            except OSError: pass
            return False

        # Concatenate the numpy arrays
        recording_np = np.concatenate(recording_data, axis=0)
        duration = len(recording_np) / samplerate
        logger.info(f"Captured {len(recording_np)} samples ({duration:.2f} seconds).")

        if duration < 0.1:
            logger.warning("Recording seems very short. May indicate silence or issue.")

        # Write the numpy array to a WAV file using the correct samplerate
        write_wav(filename, samplerate, recording_np)
        logger.info(f"User audio saved to {filename}")
        return True

    except sd.PortAudioError as e:
        logger.error(f"Sounddevice PortAudioError: {e}. Is another app using the mic? Is the device valid?")
        return False
    except IOError as e:
        logger.error(f"File IO error writing to {filename}: {e}")
        return False
    except Exception as e:
        # Catching the UnboundLocalError here if it were to persist, or other errors
        logger.error(f"An unexpected error occurred during recording: {e}", exc_info=True)
        return False
    # No finally needed as the 'with' statement handles stream closing

# --- Audio Merging ---
# ... (merge_audio_files remains the same) ...
def merge_audio_files(output_filename: str, input_files: List[str]):
    if not input_files: logger.warning("No files to merge."); return False
    if not output_filename.lower().endswith('.wav'): output_filename += ".wav"
    logger.info(f"Attempting merge: {len(input_files)} files -> '{output_filename}'")
    output_wav = None; first_file_params = None
    try:
        output_wav = wave.open(output_filename, 'wb')
        for infile_path in input_files:
            if not os.path.exists(infile_path) or os.path.getsize(infile_path) < 100: logger.warning(f"Skipping invalid/small file: {infile_path}"); continue
            try:
                with wave.open(infile_path, 'rb') as input_wav:
                    current_params = input_wav.getparams(); current_core_params = current_params[:3]
                    if first_file_params is None: logger.debug(f"Setting output params from '{os.path.basename(infile_path)}': {current_params}"); output_wav.setparams(current_params); first_file_params = current_core_params
                    else:
                        if current_core_params[2] != first_file_params[2]: logger.warning(f"Sample rate mismatch in '{os.path.basename(infile_path)}' ({current_core_params[2]} vs {first_file_params[2]}). Skipping."); continue
                        if current_core_params[:2] != first_file_params[:2]: logger.warning(f"Channel/Depth mismatch in '{os.path.basename(infile_path)}'. Skipping."); continue
                    frames = input_wav.readframes(input_wav.getnframes()); output_wav.writeframes(frames); logger.debug(f"Appended {len(frames)} bytes from '{os.path.basename(infile_path)}'")
            except wave.Error as e: logger.error(f"Error reading WAV '{infile_path}': {e}. Skipping."); continue
            except Exception as e: logger.error(f"Error processing '{infile_path}': {e}. Skipping."); continue
        if first_file_params is None: logger.error("No valid data found. Merge failed.");
        else: logger.info(f"Successfully merged audio to '{output_filename}'"); return True
    except wave.Error as e: logger.error(f"Error writing output WAV '{output_filename}': {e}")
    except Exception as e: logger.error(f"Merge error: {e}", exc_info=True)
    finally:
        if output_wav:
            try: output_wav.close()
            except Exception: pass
    if os.path.exists(output_filename) and first_file_params is None: os.remove(output_filename)
    return False


# --- Simulate Call ---
# ... (simulate_call_interactive_mic remains the same as the previous version with the input prompts) ...
def simulate_call_interactive_mic():
    app_url = input(f"Enter the Flask app URL (Default: {SIMULATOR_APP_URL}): ") or SIMULATOR_APP_URL
    location_input = input("Enter the simulated location (e.g., mumbai, delhi, default): ")
    location = location_input.lower().strip() if location_input else "default"
    if location not in LOCATION_TO_LANGUAGE and location != "unknown":
         print(f"Warning: Location '{location}' not found in config map. Using 'default'.")
         location = "default"
    audio_output_choice = input("Enable assistant audio output (TTS)? (y/N): ").lower()
    audio_output = audio_output_choice == 'y'
    call_sid = f"SIM-MIC-{uuid.uuid4()}"; call_location = location
    log_conversation_turn("SYSTEM", f"Starting Interactive Call Simulation (Location: {call_location})", call_sid=call_sid, loc=call_location)
    logger.info(f"Target App URL: {app_url}")
    lang_info = LOCATION_TO_LANGUAGE.get(call_location, LOCATION_TO_LANGUAGE['default']); sim_language_code = lang_info['lang_code']
    logger.info(f"Simulator initial language based on location '{call_location}': {sim_language_code}")
    session = requests.Session(); voice_url = f"{app_url}/voice"; current_action_url: Optional[str] = None; turn = 0
    recorded_files: List[str] = []; temp_dir_session: Optional[tempfile.TemporaryDirectory] = None
    try:
        temp_dir_session = tempfile.TemporaryDirectory(prefix=f"sim_{call_sid}_"); logger.info(f"Created temp dir: {temp_dir_session.name}")
        logger.info(f"Sending initial POST to {voice_url}"); initial_payload = {'CallSid': call_sid, 'From': '+1555SIMULATE', 'Location': call_location}
        response = session.post(voice_url, data=initial_payload); response.raise_for_status(); twiml_response = response.text; logger.debug(f"Initial TwiML:\n{twiml_response}")
        while True:
            try: root = ET.fromstring(twiml_response)
            except ET.ParseError as e: logger.error(f"Failed TwiML parse: {e}\n{twiml_response}"); break
            said_text = find_verb_text(root, 'Say')
            if said_text:
                say_lang_twilio = find_verb_attribute(root, 'Say', 'language') or map_language_code_to_twilio(sim_language_code)
                say_lang_short = next((k for k, v in LANGUAGE_MAP_TWILIO.items() if v == say_lang_twilio), sim_language_code)
                log_conversation_turn("ASSISTANT", said_text, lang=say_lang_short, call_sid=call_sid, loc=call_location)
                if audio_output:
                    tts_file_path = speak_text(said_text, lang=say_lang_short, temp_dir=temp_dir_session.name)
                    if tts_file_path: recorded_files.append(tts_file_path)
                    else: logger.warning("Failed TTS generation/playback.")
            else: logger.debug("No <Say> verb found.")
            print("-" * 30); continue_choice = input("[SIMULATOR] Press Enter to continue speaking, or type 'h' then Enter to hang up: ").lower()
            if continue_choice == 'h': log_conversation_turn("SYSTEM", "User chose to hang up.", call_sid=call_sid, loc=call_location); logger.info("User initiated hangup."); break
            gather = root.find('Gather'); hangup = root.find('Hangup')
            if hangup is not None: log_conversation_turn("SYSTEM", "<Hangup> received from application.", call_sid=call_sid, loc=call_location); break
            if gather is not None:
                gather_action = gather.get('action');
                if not gather_action: logger.error("<Gather> missing 'action'."); break
                if gather_action.startswith('/'): current_action_url = f"{app_url.rstrip('/')}{gather_action}"
                else: logger.warning(f"Gather action relative? '{gather_action}'."); current_action_url = f"{app_url.rstrip('/')}/{gather_action}"
                logger.info(f"Found <Gather>, proceeding to user audio input (Action: {current_action_url})...")
                payload: Dict[str, str] = {'CallSid': call_sid}; user_audio_path: Optional[str] = None
                if SOUNDDEVICE_AVAILABLE:
                    stop_event = threading.Event()
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"user_turn_{turn}_", dir=temp_dir_session.name) as temp_user_audio_file: user_audio_path = temp_user_audio_file.name
                        if record_audio(user_audio_path, stop_event):
                            if os.path.exists(user_audio_path) and os.path.getsize(user_audio_path) > 500:
                                 payload['RecordingUrl'] = pathlib.Path(user_audio_path).as_uri(); payload['SpeechResult'] = "[Local Audio Provided]"
                                 try:
                                     with wave.open(user_audio_path, 'rb') as wf: frames = wf.getnframes(); rate = wf.getframerate(); duration = frames / float(rate) if rate else 0; payload['RecordingDuration'] = str(int(duration))
                                 except Exception: payload['RecordingDuration'] = str(int(os.path.getsize(user_audio_path) / (SAMPLE_RATE * CHANNELS * 2)))
                                 recorded_files.append(user_audio_path); log_conversation_turn("USER", "[Recorded Audio Sent]", call_sid=call_sid, loc=call_location)
                            else: logger.warning(f"User audio file '{user_audio_path}' empty/small."); payload['SpeechResult'] = ""; payload['RecordingUrl'] = ""; payload['RecordingDuration'] = "0"; log_conversation_turn("USER", "[Silence / Recording Failed]", call_sid=call_sid, loc=call_location); os.remove(user_audio_path) if os.path.exists(user_audio_path) else None; user_audio_path = None
                        else: logger.warning("Audio recording failed."); payload['SpeechResult'] = ""; payload['RecordingUrl'] = ""; payload['RecordingDuration'] = "0"; log_conversation_turn("USER", "[Silence / Recording Failed]", call_sid=call_sid, loc=call_location); os.remove(user_audio_path) if user_audio_path and os.path.exists(user_audio_path) else None; user_audio_path = None
                    except Exception as e: logger.error(f"Error during user audio handling: {e}", exc_info=True); payload['SpeechResult'] = ""; payload['RecordingUrl'] = ""; payload['RecordingDuration'] = "0"; log_conversation_turn("USER", "[Silence / Recording Error]", call_sid=call_sid, loc=call_location); os.remove(user_audio_path) if user_audio_path and os.path.exists(user_audio_path) else None; user_audio_path = None
                else: print("[SIM] Mic disabled."); logger.warning("Sounddevice unavailable."); payload['SpeechResult'] = ""; payload['RecordingUrl'] = ""; payload['RecordingDuration'] = "0"; log_conversation_turn("USER", "[Silence / Mic Disabled]", call_sid=call_sid, loc=call_location)
                try:
                    logger.info(f"Sending POST to {current_action_url}: {payload}"); time.sleep(0.2)
                    response = session.post(current_action_url, data=payload); response.raise_for_status(); twiml_response = response.text; logger.debug(f"Received TwiML:\n{twiml_response}"); turn += 1
                except requests.exceptions.RequestException as e: logger.error(f"POST failed: {e}"); break
            else: logger.warning("No <Hangup> or <Gather> after user action."); log_conversation_turn("SYSTEM", "No further action. Ending.", call_sid=call_sid, loc=call_location); break
        log_conversation_turn("SYSTEM", "Call Simulation Ended", call_sid=call_sid, loc=call_location)
        if recorded_files:
             output_merge_file = OUTPUT_MERGED_FILENAME_TEMPLATE.format(call_sid=call_sid); wav_files_to_merge: List[str] = []; logger.info("Preparing files for merging..."); target_sample_rate: Optional[int] = None
             for fpath in recorded_files:
                 if not os.path.exists(fpath): logger.warning(f"File not found, skip merge: {fpath}"); continue
                 if fpath.lower().endswith(".mp3"):
                     wav_path = os.path.join(temp_dir_session.name, os.path.basename(fpath).replace(".mp3", ".wav"))
                     try:
                         logger.info(f"Converting MP3->WAV @{SAMPLE_RATE}Hz: '{os.path.basename(fpath)}'..."); convert_cmd = f"mpg123 -q -r {SAMPLE_RATE} --mono -w \"{wav_path}\" \"{fpath}\""; logger.debug(f"Exec: {convert_cmd}"); status = os.system(convert_cmd)
                         if status == 0 and os.path.exists(wav_path):
                             logger.info(f"Conversion OK: {wav_path}")
                             try:
                                 with wave.open(wav_path, 'rb') as wf_check: converted_rate = wf_check.getframerate()
                                 if target_sample_rate is None: target_sample_rate = converted_rate
                                 if converted_rate == target_sample_rate: wav_files_to_merge.append(wav_path)
                                 else: logger.warning(f"Converted WAV '{os.path.basename(wav_path)}' rate {converted_rate} != target {target_sample_rate}. Skip.")
                             except wave.Error as e_wave: logger.error(f"Cannot read converted WAV {wav_path}: {e_wave}. Skip.")
                         else: logger.error(f"MP3->WAV conversion failed (status: {status}). Skip file."); os.remove(wav_path) if os.path.exists(wav_path) else None
                     except Exception as e_conv: logger.error(f"Error converting MP3: {e_conv}. Skip file."); os.remove(wav_path) if os.path.exists(wav_path) else None
                 elif fpath.lower().endswith(".wav"):
                      try:
                           with wave.open(fpath, 'rb') as wf_check: current_rate = wf_check.getframerate()
                           if target_sample_rate is None: target_sample_rate = current_rate
                           if current_rate == target_sample_rate: wav_files_to_merge.append(fpath)
                           else: logger.warning(f"Existing WAV '{os.path.basename(fpath)}' rate {current_rate} != target {target_sample_rate}. Skip.")
                      except wave.Error as e_wave: logger.error(f"Cannot read existing WAV {fpath}: {e_wave}. Skip.")
                 else: logger.warning(f"Skipping unknown file type: {fpath}")
             if wav_files_to_merge: logger.info(f"Merging {len(wav_files_to_merge)} compatible WAV files."); merge_audio_files(output_merge_file, wav_files_to_merge)
             else: logger.warning("No compatible WAV files to merge.")
        else: logger.info("No audio files to merge.")
    except requests.exceptions.RequestException as e: logger.error(f"Initial connection failed: {e}")
    except Exception as e: logger.error(f"Simulation loop error: {e}", exc_info=True)
    finally:
        if temp_dir_session: logger.info(f"Cleaning up temp dir: {temp_dir_session.name}")
        try: temp_dir_session.cleanup() if temp_dir_session else None
        except OSError as e_clean: logger.error(f"Error cleaning temp dir: {e_clean}")
        logger.info("Simulator finished.")

if __name__ == "__main__":
    if not SOUNDDEVICE_AVAILABLE: print("\n---\nWARNING: Mic input disabled.\n---")
    if not TTS_AVAILABLE: print("\n---\nWARNING: Assistant audio disabled.\n---"); simulate_call_interactive_mic()
    else: simulate_call_interactive_mic()

# --- END OF FILE twilio_simulator.py ---