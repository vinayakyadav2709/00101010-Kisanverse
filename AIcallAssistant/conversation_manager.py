# --- START OF FILE conversation_manager.py ---

import logging
from config import CONVERSATION_HISTORY_LIMIT
# We don't directly call get_system_prompt here anymore, app.py does.
# from llm_handler import get_system_prompt

logger = logging.getLogger(__name__)

_call_conversations: dict[str, list] = {}
VALID_ROLES = {'user', 'assistant', 'system', 'tool'}

def get_conversation_history(call_sid):
    # If history doesn't exist, return None or empty list.
    # Initialization now happens in app.py after context is known.
    return _call_conversations.get(call_sid)

def initialize_history(call_sid, system_prompt_content):
    """Initializes history with the system prompt."""
    if call_sid not in _call_conversations:
        logger.info(f"Initializing conversation for CallSid: {call_sid}")
        _call_conversations[call_sid] = [{'role': 'system', 'content': system_prompt_content}]
    else:
        logger.warning(f"Attempted to re-initialize history for {call_sid}. Using existing.")
    return _call_conversations[call_sid]


def add_message_to_history(call_sid, role, content):
    if role not in VALID_ROLES:
        logger.error(f"Invalid role '{role}' provided.")
        return
    if call_sid not in _call_conversations:
        # This case should ideally be handled by initialization first in app.py
        logger.error(f"History not found for {call_sid} when trying to add {role} message. This might indicate a logic error.")
        # Optionally, initialize with a default/generic prompt if absolutely necessary, but it's better to ensure init happens first.
        # _call_conversations[call_sid] = [{'role': 'system', 'content': "Default system prompt - location/language unknown."}]
        return # Prevent adding message to non-existent history

    history = _call_conversations[call_sid]
    # Basic type check/conversion for content
    if not isinstance(content, str):
        try:
            content_str = str(content)
            logger.warning(f"Converted non-string content to string for role '{role}' in {call_sid}")
        except Exception:
            logger.error(f"Failed to convert content to string for role '{role}' in {call_sid}. Skipping message.")
            return
    else:
        content_str = content

    history.append({'role': role, 'content': content_str})
    logger.debug(f"Added {role} message to {call_sid}. New length: {len(history)}")
    _call_conversations[call_sid] = _trim_history(history) # Trim after adding

def _trim_history(history):
    limit = CONVERSATION_HISTORY_LIMIT
    # Ensure the system prompt (first message) is always kept
    if len(history) > (limit + 1) and history[0]['role'] == 'system':
        logger.info(f"Trimming conversation history from {len(history)} messages (keeping system prompt).")
        # Keep system prompt + last 'limit' messages
        return [history[0]] + history[-(limit):]
    elif len(history) > limit and history[0]['role'] != 'system':
         # This case is less ideal, but trim anyway if system prompt is missing/not first
         logger.warning(f"Trimming history from {len(history)} messages (no system prompt found at start).")
         return history[-limit:]
    return history # Return unchanged if within limits

def remove_last_message(call_sid, expected_role=None):
    if call_sid in _call_conversations:
        history = _call_conversations[call_sid]
        # Ensure history has more than just the system prompt before removing
        if len(history) > 1:
            if expected_role is None or history[-1]['role'] == expected_role:
                removed = history.pop()
                logger.warning(f"Removing last message (Role: {removed['role']}) for {call_sid}.")
                return True
            else:
                logger.warning(f"Did not remove last message for {call_sid}. Role '{history[-1]['role']}' != Expected '{expected_role}'.")
        elif history:
             logger.warning(f"Did not remove last message for {call_sid}. Only system prompt remains.")
    return False

def clear_conversation_history(call_sid):
    if call_sid in _call_conversations:
        logger.info(f"Clearing conversation history for CallSid: {call_sid}")
        del _call_conversations[call_sid]
        return True
    return False

# --- END OF FILE conversation_manager.py ---