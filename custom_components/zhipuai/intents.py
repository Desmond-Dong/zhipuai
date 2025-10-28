"""Simplified intent processing for 智谱清言."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from .const import DOMAIN

LOGGER = logging.getLogger(__name__)


def extract_intent_info(text: str, hass: HomeAssistant) -> Optional[Dict[str, Any]]:
    """Extract intent information from user text.

    Simplified version that only handles essential intents.
    """
    try:
        text_lower = text.lower()

        # 检查是否是取消/忽略意图
        if any(word in text_lower for word in ["算了", "取消", "不用了", "nevermind", "cancel"]):
            return {
                "intent": "nevermind",
                "text": text
            }

        # 其他意图处理可以在这里添加
        # 暂时返回None，让对话引擎处理
        return None

    except Exception as e:
        LOGGER.error("Error extracting intent: %s", e)
        return None


class SimpleIntentHandler:
    """Simplified intent handler."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the intent handler."""
        self.hass = hass

    async def handle_intent(self, intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recognized intent."""
        try:
            intent_type = intent_info.get("intent")

            if intent_type == "nevermind":
                return {
                    "success": True,
                    "message": "好的，已取消。"
                }

            return {
                "success": False,
                "error": f"Unknown intent: {intent_type}"
            }

        except Exception as e:
            LOGGER.error("Error handling intent: %s", e)
            return {
                "success": False,
                "error": str(e)
            }


def get_intent_handler(hass: HomeAssistant) -> SimpleIntentHandler:
    """Get the intent handler instance."""
    return SimpleIntentHandler(hass)


async def async_setup_intents(hass: HomeAssistant) -> None:
    """Set up intent processing."""
    LOGGER.info("Simplified intent processing setup complete")