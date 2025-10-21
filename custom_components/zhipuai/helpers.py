"""Helper functions for 智谱清言 integration."""

from __future__ import annotations

import logging
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
