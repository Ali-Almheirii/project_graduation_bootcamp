"""
LLM integration for the ERP system.

This module provides a wrapper for local LLM integration via LM Studio.
It exposes functions for sending prompts and multi-turn messages to the local
LLM and parsing its response. The wrapper connects to LM Studio running on
localhost:1234 by default.
"""

from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, Iterable, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_lm_studio_url() -> str:
    """Get the LM Studio API URL from environment or use default."""
    return os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")


class LLMError(RuntimeError):
    """Custom exception raised when an LLM call fails.
    
    This typically occurs when:
    1. LM Studio is not running
    2. LM Studio local server is not started
    3. The LLM model is not loaded
    4. Network connectivity issues
    """


def check_lm_studio_availability() -> bool:
    """Check if LM Studio is available and ready to accept requests.
    
    Returns:
        True if LM Studio is available, False otherwise.
    """
    try:
        lm_studio_url = get_lm_studio_url()
        models_url = lm_studio_url.replace("/v1/chat/completions", "/v1/models")
        response = requests.get(models_url, timeout=5)
        return response.status_code == 200
    except:
        return False


def call_lm_studio(messages: List[Dict[str, Any]]) -> str:
    """Send a list of messages to LM Studio and return the response text.

    The `messages` parameter should be a list of dictionaries in OpenAI chat format.
    Each message has a `role` (one of "user", "assistant", "system") and a `content` field.

    Raises:
        LLMError: if the request fails.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stream": False
    }
    
    lm_studio_url = get_lm_studio_url()
    
    try:
        response = requests.post(
            lm_studio_url,
            headers=headers,
            json=payload,
            timeout=60,  # Local LLM might need more time
        )
    except requests.exceptions.ConnectionError as exc:
        raise LLMError(
            f"Cannot connect to LM Studio at {lm_studio_url}. "
            f"Please ensure LM Studio is running and the local server is started. "
            f"Original error: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMError(
            f"LM Studio API request timed out after 60 seconds. "
            f"The local LLM might be processing a large request or overloaded. "
            f"Original error: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise LLMError(
            f"Failed to communicate with LM Studio API at {lm_studio_url}. "
            f"Check that LM Studio is running with the local server enabled. "
            f"Original error: {exc}"
        ) from exc

    if response.status_code != 200:
        # Attempt to extract error message from response body
        try:
            error_info = response.json()
            message = error_info.get("error", {}).get("message", response.text)
        except Exception:
            message = response.text
        
        # Provide helpful error messages based on status code
        if response.status_code == 404:
            raise LLMError(
                f"LM Studio endpoint not found (404). "
                f"Please check that LM Studio's local server is running and accessible at {lm_studio_url}. "
                f"In LM Studio, go to 'Local Server' tab and click 'Start Server'."
            )
        elif response.status_code == 500:
            raise LLMError(
                f"LM Studio internal server error (500). "
                f"This might indicate the model is not loaded or there's an issue with the LLM. "
                f"Please check LM Studio logs. Error: {message}"
            )
        elif response.status_code == 503:
            raise LLMError(
                f"LM Studio service unavailable (503). "
                f"The server might be starting up or overloaded. "
                f"Please wait a moment and try again. Error: {message}"
            )
        else:
            raise LLMError(f"LM Studio API returned status {response.status_code}: {message}")

    try:
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMError("LM Studio API returned no choices")
        
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise LLMError("LM Studio response has no content")
        
        return content.strip()
    except (KeyError, IndexError) as e:
        raise LLMError(f"Failed to parse LM Studio response: {e}")


def call_gemini(messages: List[Dict[str, Any]]) -> str:
    """Compatibility wrapper - converts Gemini format to OpenAI format and calls LM Studio.
    
    Converts Gemini's format with 'parts' to standard OpenAI chat format.
    """
    # Convert Gemini format to OpenAI format
    openai_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        # Map Gemini roles to OpenAI roles
        if role == "model":
            role = "assistant"
        
        # Extract text from parts
        parts = msg.get("parts", [])
        if parts and isinstance(parts, list) and len(parts) > 0:
            content = parts[0].get("text", "")
        else:
            content = str(msg.get("content", ""))
        
        openai_messages.append({
            "role": role,
            "content": content
        })
    
    return call_lm_studio(openai_messages)


def call_gemini_prompt(prompt: str) -> str:
    """Convenience wrapper for calling LM Studio with a single prompt.

    The prompt is wrapped into a single user message and sent to LM Studio.
    Returns the plain text response.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    return call_lm_studio(messages)
