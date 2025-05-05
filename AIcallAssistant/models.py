# models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class STTResult(BaseModel):
    text: str = Field(..., description="Transcribed text.")
    language: str = Field(..., description="Detected language code (e.g., 'en', 'hi').")
    confidence: Optional[float] = Field(None, description="Overall transcription confidence (0.0 to 1.0). Often unavailable.")
    is_reliable: bool = Field(True, description="Flag indicating if the transcription is considered reliable (e.g., not empty).")

class LLMToolCall(BaseModel):
    tool_name: str = Field(..., description="The name of the tool/function to be called.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")

class ToolResult(BaseModel):
    success: bool = Field(..., description="Whether the tool execution was successful.")
    data: Optional[Dict[str, Any]] = Field(None, description="Data returned by the tool on success.")
    error: Optional[str] = Field(None, description="Error message on failure.")