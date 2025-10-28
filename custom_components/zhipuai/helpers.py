"""Helper functions for 智谱清言 integration."""

from __future__ import annotations

import base64
import io
import json
import logging
import wave
from typing import Any

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
