"""Helper functions for 智谱清言 integration."""

from __future__ import annotations

import base64
import io
import json
import logging
import wave
from contextlib import suppress
from typing import Any

from homeassistant.exceptions import HomeAssistantError

_LOGGER = logging.getLogger(__name__)


def format_error_message(error: Exception) -> str:
    """Format error message for user display."""
    error_msg = str(error)

    # Check for common error types
    if "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
        return "API密钥无效或已过期，请检查配置"

    if "rate" in error_msg.lower() or "limit" in error_msg.lower():
        return "请求过于频繁，请稍后再试"

    if "timeout" in error_msg.lower():
        return "请求超时，请检查网络连接"

    if "network" in error_msg.lower() or "connection" in error_msg.lower():
        return "网络连接失败，请检查网络设置"

    # Return original message
    return f"发生错误: {error_msg}"


def truncate_history(
    messages: list[dict[str, Any]], max_messages: int
) -> list[dict[str, Any]]:
    """Truncate message history to max_messages."""
    if len(messages) <= max_messages:
        return messages

    # Always keep system message if it exists
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    other_messages = [msg for msg in messages if msg.get("role") != "system"]

    # Keep the most recent messages
    if len(other_messages) > max_messages - len(system_messages):
        other_messages = other_messages[-(max_messages - len(system_messages)):]

    return system_messages + other_messages


def decode_base64_audio(base64_data: str, sample_rate: int = 24000) -> bytes:
    """Decode base64 encoded audio data to WAV format.

    Args:
        base64_data: Base64 encoded audio data
        sample_rate: Audio sample rate (default: 24000 Hz for 智谱AI TTS)

    Returns:
        WAV format audio data as bytes
    """
    try:
        # Decode base64 to raw bytes
        raw_audio_data = base64.b64decode(base64_data)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio_data)

        return wav_buffer.getvalue()
    except Exception as exc:
        _LOGGER.error("Failed to decode base64 audio: %s", exc)
        raise ValueError(f"音频解码失败: {exc}") from exc


def parse_streaming_response(response_text: str) -> list[str]:
    """Parse streaming TTS response and extract audio data.

    Args:
        response_text: The streaming response text containing data: lines

    Returns:
        List of base64 encoded audio data strings
    """
    audio_chunks = []
    lines = response_text.strip().split('\n')

    for line in lines:
        if line.startswith('data: '):
            try:
                data_str = line[6:]  # Remove 'data: ' prefix
                data_dict = json.loads(data_str)  # Parse JSON string

                # Extract audio content from streaming response
                if 'choices' in data_dict and len(data_dict['choices']) > 0:
                    choice = data_dict['choices'][0]
                    if 'delta' in choice and 'content' in choice['delta']:
                        audio_content = choice['delta']['content']
                        if audio_content and audio_content != "":
                            audio_chunks.append(audio_content)
            except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                _LOGGER.warning("Failed to parse streaming audio chunk: %s", exc)
                continue

    return audio_chunks


def combine_audio_chunks(audio_chunks: list[str]) -> str:
    """Combine multiple base64 audio chunks into a single base64 string.

    Args:
        audio_chunks: List of base64 encoded audio chunks

    Returns:
        Combined base64 encoded audio data
    """
    try:
        # Decode all chunks and combine the raw audio data
        combined_raw = b""
        for chunk in audio_chunks:
            raw_data = base64.b64decode(chunk)
            combined_raw += raw_data

        # Re-encode the combined data
        return base64.b64encode(combined_raw).decode('utf-8')
    except Exception as exc:
        _LOGGER.error("Failed to combine audio chunks: %s", exc)
        raise ValueError(f"音频合并失败: {exc}") from exc


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generate a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file.
    """
    _LOGGER.info("Converting audio data (%d bytes) from MIME type: %s", len(audio_data), mime_type)

    try:
        parameters = _parse_audio_mime_type(mime_type)
        _LOGGER.info("Parsed audio parameters: %s", parameters)
    except Exception as exc:
        _LOGGER.warning("Failed to parse MIME type %s: %s, using defaults", mime_type, exc)
        # If MIME type parsing fails, use default parameters
        parameters = {"bits_per_sample": 16, "rate": 16000}

    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(parameters["bits_per_sample"] // 8)
            wf.setframerate(parameters["rate"])
            wf.writeframes(audio_data)

        result = wav_buffer.getvalue()
        _LOGGER.info("Successfully created WAV file (%d bytes)", len(result))
        return result
    except Exception as exc:
        _LOGGER.error("Failed to create WAV file: %s", exc, exc_info=True)
        raise


def _parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """Parse bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    if not mime_type.startswith("audio/L"):
        _LOGGER.warning("Received unexpected MIME type %s", mime_type)
        # Don't raise error, use default values instead
        return {"bits_per_sample": 16, "rate": 16000}

    bits_per_sample = 16
    rate = 16000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts[1:]:  # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            # Handle cases like "rate=" with no value or non-integer value and keep rate as default
            with suppress(ValueError, IndexError):
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
        elif param.startswith("audio/L"):
            # Keep bits_per_sample as default if conversion fails
            with suppress(ValueError, IndexError):
                bits_per_sample = int(param.split("L", 1)[1])

    return {"bits_per_sample": bits_per_sample, "rate": rate}
